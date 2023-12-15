import torch
from einops import rearrange
import copy
from hidden_state_cache.utils import get_cache_type
import math
import time


class LowRankQKV:
    def __init__(self, model, rank, weight_file=None, cache_type="full"):
        self.n_layers = model.config.n_layers
        self.d_model = model.config.d_model
        self.rank = rank
        self.n_heads = model.config.n_heads
        if cache_type != "full":
            raise ValueError("FastQKV only supports full cache")

        self.norms = torch.nn.ModuleList()
        self.norm_scale_inv = []
        self.encoders = []

        self.modified_Wv = []
        self.modified_Wk = []
        self.Wq = []
        self.out_proj = []

        if not weight_file:
            raise ValueError("weight_file must be provided")
        weights = torch.load(weight_file)
        with torch.no_grad():
            for n_layer in range(self.n_layers):
                Vh = weights[n_layer].to(torch.float16)

                Vh_lowrank = Vh[: self.rank, :]  # shape rank x 4096
                # rows of Vh are the important part
                encoder_weight = Vh_lowrank  # shape rank x 4096

                encoder = torch.nn.Linear(
                    in_features=model.config.d_model,
                    out_features=self.rank,
                    bias=False,
                )
                encoder.weight = torch.nn.Parameter(encoder_weight)
                self.encoders.append(encoder)
                # encoder will right multiply by Vh_lr^T

                norm_1 = copy.deepcopy(model.transformer.blocks[n_layer].norm_1)
                norm_1_weight = norm_1.weight
                norm_1_sign = torch.sign(norm_1_weight)
                norm_1_unsigned_clamp = torch.clamp(torch.abs(norm_1_weight), 1e-4, 1e4)

                self.norm_scale_inv.append(norm_1_sign / norm_1_unsigned_clamp)

                norm_1.weight = torch.nn.Parameter(torch.ones_like(norm_1_weight))
                self.norms.append(norm_1)

                Wq, Wk, Wv = model.transformer.blocks[n_layer].attn.Wqkv.weight.chunk(
                    3, dim=0
                )
                # should be half precision automatically
                Wk = torch.matmul(Wk, torch.diag(norm_1_weight))
                modified_Wk_weight = torch.matmul(Wk, Vh_lowrank.t())
                modified_Wk_weight = rearrange(
                    modified_Wk_weight, "(h dd) k -> h dd k", h=self.n_heads
                ).unsqueeze(0)
                self.modified_Wk.append(modified_Wk_weight)  # shape 1 x 32 x 128 x rank

                # encoded_state * Vh_lr * norm_1.weight * Wv^T
                # or transpose of Wv * norm_1.weight * Vh_lr.t()

                Wv = torch.matmul(Wv, torch.diag(norm_1_weight))
                modified_Wv_weight = torch.matmul(Wv, Vh_lowrank.t())
                modified_Wv_weight = rearrange(
                    modified_Wv_weight, "(h dd) k -> h dd k", h=self.n_heads
                ).unsqueeze(0)
                self.modified_Wv.append(modified_Wv_weight)  # shape 1 x 32 x 128 x rank

                self.Wq.append(Wq)

                # self.Wo.append(model.transformer.blocks[n_layer].attn.out_proj.weight)
                self.out_proj.append(model.transformer.blocks[n_layer].attn.out_proj)

    def encode(self, layer_idx, hidden_state):
        """
        input shape b, s, d
        output shape b, s, k
        """
        normed_hidden_state = self.norms[layer_idx](hidden_state)

        if self.encoders:
            output = self.encoders[layer_idx](normed_hidden_state)
        else:
            assert self.rank == self.d_model
            output = normed_hidden_state
        return output

    def get_queries(self, layer_idx, normed_query_hidden_states):
        """
        input shape b, q, d
        output shape b, h, q, k
        """
        queries_1_time = time.time()
        old_queries = torch.matmul(normed_query_hidden_states, self.Wq[layer_idx].t())
        # print("queries_1_time: ", time.time() - queries_1_time)
        queries_2_time = time.time()
        old_queries = rearrange(old_queries, "b q (h dd) -> b h q dd", h=self.n_heads)
        # print("queries_2_time: ", time.time() - queries_2_time)
        queries_3_time = time.time()
        modified_queries = torch.matmul(old_queries, self.modified_Wk[layer_idx])
        # print("queries_3_time: ", time.time() - queries_3_time)
        queries_4_time = time.time()
        modified_queries = rearrange(
            modified_queries, "b h q dd -> b q (h dd)", h=self.n_heads
        )
        # print("queries_4_time: ", time.time() - queries_4_time)

        # b, h, q, dd * b, h, dd, k -> b, h, q, k -> b, q, (h k)

        return modified_queries

    def compute_qkv(
        self,
        layer_idx,
        past_encoded_hidden_states,
        normed_query_hidden_states,
    ):
        encoders_time = time.time()
        if self.encoders:
            unscaled_query_hidden_states = torch.matmul(
                normed_query_hidden_states, torch.diag(self.norm_scale_inv[layer_idx])
            )
            # unscaled_query_hidden_states = normed_query_hidden_states
            # need to unscale, since the scale is built into the encoders...
            # mayber later just make this faster by feeding in the unnormed hidden states lmao
            current_encoded_hidden_states = self.encoders[layer_idx](
                unscaled_query_hidden_states
            )
        else:
            assert self.rank == self.d_model
            current_encoded_hidden_states = normed_query_hidden_states

        # print("encoders_time: ", time.time() - encoders_time)
        # b,q,d -> b,q,k
        # don't need to norm it again, because it is already normed

        if past_encoded_hidden_states is not None:
            keys = torch.concat(
                (past_encoded_hidden_states, current_encoded_hidden_states), dim=1
            )
            # b,s,k so dim=1
        else:
            keys = current_encoded_hidden_states

        # shape b,s,k
        # we don't unsqueeze because attention_func will do it for us

        # repeat across 32 heads
        # Add a new dimension of size 1
        get_queries_time = time.time()
        modified_queries = self.get_queries(layer_idx, normed_query_hidden_states)
        # print("get_queries_time: ", time.time() - get_queries_time)
        # shape b, q, (h k)

        # shapes b, q, (h k) and b, s, k and b, s, k
        return modified_queries, keys, keys

    def compute_output(self, layer_idx, attn_output):
        # attn_output is shape b, q, (h k) (already multiplied by v, rearranged)
        # modified_Wv is shape 1, h, dd, k
        # out_proj is shape d, d
        # output is shape b, h, q, d

        # print("attn_output shape: ", attn_output.shape)
        # print("modified_Wv shape: ", self.modified_Wv[layer_idx].shape)

        attn_output = rearrange(attn_output, "b q (h k) -> b h q k", h=self.n_heads)

        modified_attn_output = torch.matmul(
            attn_output, self.modified_Wv[layer_idx].transpose(-1, -2)
        )
        # now shape b,h,q,dd
        modified_attn_output = rearrange(modified_attn_output, "b h q dd -> b q (h dd)")
        return self.out_proj[layer_idx](modified_attn_output)


class FullRankQKV(LowRankQKV):
    """
    Special case when encoder is identity!
    """

    def __init__(self, model, rank, weight_file=None, cache_type="full"):
        self.n_layers = model.config.n_layers
        self.d_model = model.config.d_model
        self.rank = rank
        self.n_heads = model.config.n_heads
        if cache_type != "full":
            raise ValueError("QKV only supports full cache")

        self.norms = torch.nn.ModuleList()

        self.modified_Wv = []
        self.modified_Wk = []
        self.Wq = []
        self.out_proj = []
        self.encoders = None

        for n_layer in range(self.n_layers):
            self.norm_1 = model.transformer.blocks[n_layer].norm_1
            self.norms.append(copy.deepcopy(self.norm_1))

            Wq, Wk, Wv = model.transformer.blocks[n_layer].attn.Wqkv.weight.chunk(
                3, dim=0
            )
            # should be half precision automatically
            modified_Wk_weight = rearrange(
                Wk, "(h dd) k -> h dd k", h=self.n_heads
            ).unsqueeze(0)
            self.modified_Wk.append(modified_Wk_weight)  # shape 1 x 32 x 128 x rank

            modified_Wv_weight = rearrange(
                Wv, "(h dd) k -> h dd k", h=self.n_heads
            ).unsqueeze(0)
            self.modified_Wv.append(modified_Wv_weight)  # shape 1 x 32 x 128 x rank

            self.Wq.append(Wq)

            # self.Wo.append(model.transformer.blocks[n_layer].attn.out_proj.weight)
            self.out_proj.append(model.transformer.blocks[n_layer].attn.out_proj)
