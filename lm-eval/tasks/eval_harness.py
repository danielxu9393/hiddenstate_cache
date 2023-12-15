from functools import partial

import os
import transformers
from lm_eval.base import LM
from tqdm import tqdm
import numpy as np
import torch

from tasks.util import sample_batch, shrink_seq
import multiprocessing
import ftfy

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from hidden_state_cache.utils import load
from hidden_state_cache.hidden_state_cache.hidden_state_cache import HiddenStateCache

class EvalHarnessLM(LM):
    def greedy_until(self, requests):
        raise Exception("unimplemented")

    def loglikelihood_rolling(self, requests):
        raise Exception("unimplemented")

    def __init__(self, model, tokenizer, args, low_rank_qkv=None):
        super().__init__()
        self.model = model
        self.low_rank_qkv = low_rank_qkv
        self.tokenizer = tokenizer
        self.cache_type = args.cache_type
        self.cache_args = args.cache_args
        self.start_size = args.start_size
        self.recent_size = args.recent_size
        self.cache_size = args.cache_size
        self.seq_dim = args.seq_dim
        self.enable_hscache = args.enable_hscache
        self.use_kv = args.use_kv
        self.baseline = args.baseline

    def loglikelihood(self, requests):
        output = []

        with torch.no_grad():
            for context, continuation in tqdm(requests):
                context = ftfy.fix_text(context, normalization="NFKC")
                continuation = ftfy.fix_text(continuation, normalization="NFKC")
                print("continuation: ", continuation)
                ctx_tokens = self.tokenizer(
                    context, add_special_tokens=False, return_tensors="pt"
                ).input_ids.to(self.model.device)
                cont_tokens = self.tokenizer(
                    continuation, add_special_tokens=False, return_tensors="pt"
                ).input_ids.to(self.model.device)
                print("cont_tokens: ", cont_tokens)

                past_key_values = None
                if self.enable_hscache:
                    hidden_state_cache = HiddenStateCache(
                        batch_size=1,
                        n_layers=self.model.config.n_layers,
                        keep_hidden_layer_idx=None,
                        cache_type=self.cache_type,
                        cache_args=self.cache_args,
                        start_size=self.start_size,
                        recent_size=self.recent_size,
                        cache_size=self.cache_size,
                        seq_dim=self.seq_dim,
                        baseline=self.baseline,
                        low_rank_qkv=self.low_rank_qkv,
                    )
                else:
                    hidden_state_cache = None

                cont_logits = []
                outputs_ctx = self.model(
                    ctx_tokens[:, :-1],
                    hidden_state_cache=hidden_state_cache,
                    past_key_values=past_key_values,
                    use_cache=self.use_kv,
                    use_hscache=self.enable_hscache,
                    output_hidden_states=True,
                )
                if hidden_state_cache is not None:
                    hidden_state_cache.add_to_cache(outputs_ctx.hidden_states)
                if self.use_kv:
                    past_key_values = outputs_ctx.past_key_values

                # last token of context fed in separately to ensure hscache is sparse for all continuation predictions
                # cont_tokens_predecessor is last context token, and all but last continuation token
                cont_tokens_predecessor = torch.cat(
                    (ctx_tokens[:, -1:], cont_tokens[:, :-1]), dim=1
                )
                assert cont_tokens_predecessor.shape == cont_tokens.shape

                # logits = outputs.logits.log_softmax(dim=-1)
                # logits = logits[:, -1, :].unsqueeze(1)
                # cont_logits.append(logits[:, -1, :].unsqueeze(1))

                for i in range(cont_tokens_predecessor.size(1)):
                    outputs = self.model(
                        cont_tokens_predecessor[:, i : i + 1],
                        hidden_state_cache=hidden_state_cache,
                        past_key_values=past_key_values,
                        use_cache=self.use_kv,
                        use_hscache=self.enable_hscache,
                        output_hidden_states=True,
                    )
                    if hidden_state_cache is not None:
                        hidden_state_cache.add_to_cache(outputs.hidden_states)
                    if self.use_kv:
                        past_key_values = outputs.past_key_values

                    logits = outputs.logits.log_softmax(dim=-1)
                    cont_logits.append(logits)

                for logits in cont_logits:
                    assert logits.shape == (1, 1, self.model.config.vocab_size)
                cont_logits = torch.cat(cont_logits, dim=1)

                values, indices = cont_logits.squeeze(0).topk(dim=-1, k=1)
                gold_indices = cont_tokens.squeeze(0)
                values = values.squeeze(1)
                indices = indices.squeeze(1)

                ### Note: it seems like continuation always ends with the end token so that is taken care of!!

                correct = True
                total_logprob = 0
                for i in range(len(gold_indices)):
                    if gold_indices[i] != indices[i]:
                        correct = False
                    logprob = cont_logits[0, i, gold_indices[i]]
                    total_logprob += logprob

                total_logprob = total_logprob.item()

                output.append((total_logprob, correct))
                print((total_logprob, correct))
                ### Shouldn't this be positive log likelihood???
                ### (In their code they negate it twice lmfaoo)
                ### Also why did they divide by length? Because in the lm_eval code, they will divide for you
                ### for the acc_norm metric

        return output
