import argparse
import json
import os
import sys
import importlib

from lm_eval import evaluator, tasks
from tasks import EvalHarnessLM, EvalHarnessAdaptor

from hidden_state_cache.utils import load, load_with_callback
from hidden_state_cache.hidden_state_cache.low_rank_qkv import LowRankQKV, FullRankQKV


def json_to_key(obj):
    return json.dumps(obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    parser.add_argument("--output-file", type=str, default="output.jsonl")
    parser.add_argument("--task-name", type=str, default="hellaswag")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--num-data", type=int, default=None)

    parser.add_argument("--enable-hscache", action="store_true")
    parser.add_argument("--use-kv", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    ### whether should opt to not replace evicted hidden states
    parser.add_argument("--model-path", type=str, default="../models/mpt-7b-hscache")
    parser.add_argument("--cache-type", type=str, default="full")
    parser.add_argument("--cache-args", type=int, default=None)

    parser.add_argument("--start-size", type=int, default=4)
    parser.add_argument("--recent-size", type=int, default=20)
    parser.add_argument("--cache-size", type=int, default=2048)

    parser.add_argument("--enable-lowrank", action="store_true")

    parser.add_argument(
        "--weight_file", type=str, default="../models/low_rank_weights_mpt.pth"
    )
    parser.add_argument("--rank", type=int, default=4096)

    args = parser.parse_args()

    # def config_mha_attn_func(config):
    #     config.attn_config["attn_impl"] = "torch"
    # model, tokenizer = load_with_callback(args.model_path, config_mha_attn_func)
    # assert model.config.attn_config["attn_impl"] == "torch"

    model, tokenizer = load(args.model_path)

    print("model loaded")

    model.half().eval().cuda()

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
        args.seq_dim = 1
    else:
        raise ValueError(f"got {model.config.model_type}")

    lm = EvalHarnessLM(model, tokenizer, args, low_rank_qkv)

    results = evaluator.evaluate(
        lm,
        tasks.get_task_dict(
            [
                args.task_name
                # "lambada_openai",
                # "piqa",
                # "hellaswag",
                # "winogrande",
                # "mathqa",
                # "pubmedqa",
                # "boolq",
                # "cb",
                # "copa",
                # "multirc",
                # "record",
                # "wic",
                # "wsc",
            ]
        ),
        False,
        args.num_fewshot,
        args.num_data,
    )

    dumped = json.dumps(results, indent=2)
    print(dumped)

    with open(args.output_file, "w") as f:
        f.write(dumped)
