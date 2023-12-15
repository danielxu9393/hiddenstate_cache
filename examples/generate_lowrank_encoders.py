import torch
import argparse
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from hidden_state_cache.kv_cache import StartRecentKVCache
from hidden_state_cache.utils import parse_args, load
from hidden_state_cache.hidden_state_cache.hidden_state_cache import HiddenStateCache
import time
from einops import rearrange


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    parser.add_argument("--model-path", type=str, default="../models/mpt-7b-hscache")
    parser.add_argument(
        "--output-file", type=str, default="../models/low_rank_weights_mpt.pth"
    )
    parser.add_argument("--modify", action="store_true")
    # modify Wkv by merging the LN gamma into it
    parser.add_argument("--norm", action="store_true")
    # use Frobenius norm to normalize each head

    args = parser.parse_args()

    device = "cuda"

    model, tokenizer = load(args.model_path)

    with torch.no_grad():
        save_matrices = []
        for n_layer in range(model.config.n_layers):
            # for n_layer in range(1):
            torch.cuda.empty_cache()
            print("Layer: ", n_layer)

            norm_1 = model.transformer.blocks[n_layer].norm_1
            # print("norm_1: ", norm_1.weight, norm_1.bias)
            norm_mean = torch.mean(norm_1.weight, dim=0).item()
            norm_std = torch.std(norm_1.weight, dim=0).item()
            print("norm_1 mean, std: ", norm_mean, norm_std)

            # yay it doesn't have a bias!! Also the weights are all very close to each other!!
            # The fact that it doesn't have a bias is very nice... So if LN is of the form
            # norm(x) * diag(gamma), then the whole computation is
            # x is nxd: norm(x) * diag(gamma) * Wqkv^T... Can diag(gamma) be fused into Wqkv?
            # (Pytorch automatically transposes Wqkv to handle batch dimension!)
            Wq, Wk, Wv = (
                model.transformer.blocks[n_layer]
                .attn.Wqkv.weight.float()
                .chunk(3, dim=0)
            )
            Wo = model.transformer.blocks[n_layer].attn.out_proj.weight.float()
            # shape 2*4096 x 4096, dimension of weight is out_sizexin_size
            # .float() just casts it to fp32

            if args.modify:
                Wq = torch.matmul(Wq, torch.diag(norm_1.weight.float()))
                Wk = torch.matmul(Wk, torch.diag(norm_1.weight.float()))
                Wv = torch.matmul(Wv, torch.diag(norm_1.weight.float()))

            h = 32
            q = 4096

            WqT_split = rearrange(Wq, "d (h dd) -> h d dd", h=h)
            # be careful with the order of the dimensions, make sure no transposing stuff?
            Wk_split = rearrange(Wk, "d (h dd) -> h dd d", h=h)
            WqTWk = torch.matmul(WqT_split, Wk_split)  # shape h, d, d

            # WoT_split = rearrange(Wo, "d (h dd) -> h d dd", h=h)
            Wo_split = rearrange(Wo, "d (h dd) -> h d dd", h=h)
            # TODO: I think it should be Wo and not WoT!!!
            Wv_split = rearrange(Wv, "(h dd) d -> h dd d", h=h)
            # WoTWv = torch.matmul(WoT_split, Wv_split)  # shape h, d, d
            WoWv = torch.matmul(Wo_split, Wv_split)  # shape h, d, d

            W_combined = torch.cat([WqTWk, WoWv], dim=0)  # shape 2h, d, d
            del WqTWk, WoWv
            del WqT_split, Wk_split, Wo_split, Wv_split
            del Wq, Wk, Wo, Wv

            if args.norm:
                combined_norm = W_combined.norm(dim=(1, 2), keepdim=True)
                W_combined = W_combined / combined_norm
            W_combined = rearrange(W_combined, "h d d2 -> (h d) d2")

            # q: should we even norm across heads? I think so because its all relative!
            # like norming just means we care about each head equally
            # and we care about k,v equally...

            U, S, Vh = torch.svd_lowrank(W_combined, q=q)
            # shapes 2hdxd, d, dxd

            # we only care about encoder Vh???
            # We don't even need to unnorm it!!
            # Wq^TWk = Wq^T * (Wk * Vh^T) * Vh
            # just store Vh is dxd!!!
            # and we can adjust the rank!!
            # USVh = norm_1.weight^T Wq^T Wk norm_1.weight
            # U_lr S_lr Vh_lr = norm_1.weight^T Wq^T Wk norm_1.weight Vh_lr^T Vh_lr = norm_1.weight^T Wq^T Wk Vh_lr'^T Vh_lr' norm_1.weight
            # If we have Vh_lr' = Vh_lr / norm_1.weight!!
            # So we should divide it!
            # Actually I don't wanna denorm it, so still norm_1.weight^T Wq^T Wk norm_1.weight Vh_lr^T Vh_lr
            # So need to multiply Wk, Wv by norm_1.weight

            # We combined Wk * norm_1.weight * Vh_lr^T

            # now look at ov:
            # U2SVh = Wo^T Wv norm_1.weight
            # U2_lr S_lr Vh_lr = Wo^T Wv norm_1.weight Vh_lr^T Vh_lr
            # We combine Wv * norm_1.weight * Vh_lr^T

            # Computation is v = x * norm_1.weight * Wv^T
            # context = attn_scores * v
            # b, h, q, s * 1, h, s, dd = b, h, q, dd (then rearrange to b, q, (h dd))
            #
            # output = context * Wo^T
            # attn_scores * x * norm_1.weight * Wv^T * Wo^T

            # new computation is:

            # attn_scores * x * Vh_lr^T * Vh_lr * norm_1.weight * Wv^T * Wo^T
            # attn_scores * encoded_state * Vh_lr * norm_1.weight * Wv^T * Wo^T
            # combine Vh_lr * norm_1.weight * Wv^T, right multiply???

            # norm_1_weight_inv = torch.diag(1 / norm_1.weight.float())
            # Vh_denormed = torch.matmul(Vh, norm_1_weight_inv)
            # we don't denorm it lmao... because the norm_1.weight sometimes has 0s...

            for i in range(13):
                print("Singular Value pow2 ", i, ": ", S[2**i - 1].item())
            print("Singular Value 3000: ", S[3000].item())
            print("Singular Value 3500: ", S[3500].item())
            print("Singular Value 3750: ", S[3750].item())
            print("Singular Value 4000: ", S[4000].item())
            del U, S
            # save_matrices.append(Vh_denormed)
            save_matrices.append(Vh)

        if args.output_file is not None:
            torch.save(save_matrices, args.output_file)
            print("Saved to ", args.output_file)
