import argparse
import json, tqdm
import torch
import copy


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from hidden_state_cache.utils import load
from hidden_state_cache.hidden_state_cache.hidden_state_cache import HiddenStateCache


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    parser.add_argument("--input-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--enable-hscache", action="store_true")
    parser.add_argument("--model-path", type=str, default="../models/mpt-7b-hscache")
    parser.add_argument("--cache-type", type=str, default="full")

    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=20)
    parser.add_argument("--cache_size", type=int, default=512)

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    model_path = args.model_path
    cache_type = args.cache_type
    enable_hscache = args.enable_hscache

    model, tokenizer = load(args.model_path)

    model.half().eval().cuda()

    requests = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip() != "":
                requests.append(json.loads(line))

    results = []
    with torch.no_grad():
        for request in tqdm.tqdm(requests):
            result = {"request": request, "result": {}}
            prompt = request["prompt"]
            input_ids = tokenizer(
                prompt, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(model.device)

            outputs = model(
                input_ids,
                hidden_state_cache=None,
                use_cache=True,
                use_hscache=enable_hscache,
                output_hidden_states=True,
            )
            logits = outputs.logits.log_softmax(dim=-1)

            values, indices = logits.squeeze(0).topk(dim=-1, k=1)
            tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
            # seems like batch size always 1

            gold_indices = input_ids[:, 1:]  # skip first
            logprobs = [None] + torch.gather(
                logits, -1, gold_indices.unsqueeze(-1)
            ).squeeze(-1).squeeze(0).detach().cpu().tolist()

            top_logprobs = [None] + [
                {tokenizer.convert_ids_to_tokens(i.item()): v.item()}
                for v, i in zip(values.squeeze(-1), indices.squeeze(-1))
            ]

            result["result"] = {
                "choices": [
                    {
                        "text": prompt,
                        "logprobs": {
                            "tokens": tokens,
                            "token_logprobs": logprobs,
                            "top_logprobs": top_logprobs,
                            "text_offset": [],
                        },
                        "finish_reason": "length",
                    }
                ],
                "request_time": {"batch_time": 0, "batch_size": 1},
            }

            results.append(result)

    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
