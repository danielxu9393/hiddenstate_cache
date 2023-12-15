import torch
import argparse
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss

# from hidden_state_cache.kv_cache import StartRecentKVCache
from hidden_state_cache.utils import parse_args, load
from hidden_state_cache.hidden_state_cache.hidden_state_cache import HiddenStateCache
from hidden_state_cache.hidden_state_cache.low_rank_qkv import LowRankQKV, FullRankQKV
import time

import torch.autograd.profiler as profiler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    parser.add_argument("--ppl-file", type=str, default="ppl.txt")
    parser.add_argument("--log-file", type=str, default="log.txt")
    parser.add_argument("--task", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument(
        "--split", type=str, default="test", choices=["validation", "test"]
    )

    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-eval-tokens", type=int, default=None)

    parser.add_argument("--enable-hscache", action="store_true")
    parser.add_argument("--enable-lowrank", action="store_true")
    parser.add_argument("--use-kv", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    ### whether should opt to not replace evicted hidden states
    parser.add_argument("--model-path", type=str, default="../models/mpt-7b-hscache")
    parser.add_argument("--cache-type", type=str, default="full")
    parser.add_argument("--cache-args", type=int, default=None)

    parser.add_argument("--start-size", type=int, default=4)
    parser.add_argument("--recent-size", type=int, default=5)
    parser.add_argument("--cache-size", type=int, default=512)

    parser.add_argument(
        "--weight_file", type=str, default="../models/low_rank_weights_mpt.pth"
    )
    parser.add_argument("--rank", type=int, default=4096)

    args = parser.parse_args()

    device = "cuda"

    if args.task == "wikitext-2-raw-v1":
        data = load_dataset(args.dataset_name, args.task, split=args.split)
    else:
        data = load_dataset(args.dataset_name, split=args.split)

    model, tokenizer = load(args.model_path)

    nlls = []
    loss_fn = CrossEntropyLoss(reduction="none")
    past_key_values = None

    f = open(args.log_file, "w")

    if args.enable_lowrank:
        print(f"Using rank {args.rank} QKV")
        assert args.rank <= 4096
        if args.rank == 4096:
            low_rank_qkv = FullRankQKV(
                model,
                args.rank,
            )
        else:
            low_rank_qkv = LowRankQKV(
                model,
                args.rank,
                args.weight_file,
            )
    else:
        low_rank_qkv = None

    if "mpt" in model.config.model_type:
        seq_dim = 1
        k_seq_dim = 3  # b h d s
        v_seq_dim = 2  # b h s d
    else:
        raise ValueError(f"got {model.config.model_type}")

    start_time = time.time()

    num_eval_tokens = 0
    for text in data["text"][: args.num_samples]:
        encodings = tokenizer(text, return_tensors="pt")

        seq_len = encodings.input_ids.size(1)

        batch_size = encodings.input_ids.size(0)
        if args.enable_hscache:
            hidden_state_cache = HiddenStateCache(
                batch_size=batch_size,
                n_layers=model.config.n_layers,
                keep_hidden_layer_idx=None,
                start_size=args.start_size,
                recent_size=args.recent_size,
                cache_size=args.cache_size,
                seq_dim=seq_dim,
                cache_type=args.cache_type,
                low_rank_qkv=low_rank_qkv,
            )
        else:
            hidden_state_cache = None

        # if args.use_kv:
        #     kv_cache = StartRecentKVCache(
        #         start_size=args.start_size,
        #         recent_size=args.cache_size,
        #         k_seq_dim=k_seq_dim,
        #         v_seq_dim=v_seq_dim,
        #     )

        pbar = tqdm(range(0, seq_len - 1))

        for idx in pbar:
            loop_time = time.time()
            input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
            with torch.no_grad():
                model_time = time.time()
                outputs = model(
                    input_ids,
                    past_key_values=past_key_values,
                    hidden_state_cache=hidden_state_cache,
                    use_cache=args.use_kv,
                    use_hscache=args.enable_hscache,
                    output_hidden_states=True,
                )

                logits = outputs.logits.view(-1, model.config.vocab_size)
                past_key_values = outputs.past_key_values

                label = (
                    encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
                )

                neg_log_likelihood = loss_fn(logits, label)

                cache_time = time.time()
                if hidden_state_cache is not None:  # update the cache!
                    hidden_state_cache.add_to_cache(outputs.hidden_states)
                if args.use_kv:
                    past_key_values = kv_cache(past_key_values)

            nlls.append(neg_log_likelihood)
            pbar.set_description(
                f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
            )
            print(neg_log_likelihood.item(), file=f, flush=True)
            num_eval_tokens += 1
            if (
                args.num_eval_tokens is not None
                and num_eval_tokens >= args.num_eval_tokens
            ):
                break
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            break

    f.close()

    end_time = time.time()
    elapsed_time = end_time - start_time

    ppl = torch.exp(torch.stack(nlls).mean())
    print(ppl.item())
    with open(args.ppl_file, "w") as f:
        f.write(f"ppl: {ppl.item()}\n")
        f.write(f"num_eval_tokens: {num_eval_tokens}\n")
        f.write(f"elapsed_time: {elapsed_time}\n")
