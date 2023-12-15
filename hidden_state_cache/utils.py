import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
)
import os.path as osp
import ssl
import urllib.request
import os
import json


def get_cache_type(cache_type, n_layers, cache_args=None):
    if cache_type == "full":
        keep_hidden_layer_idx = [i for i in range(n_layers)]
    elif cache_type == "half":
        keep_hidden_layer_idx = [_ for _ in range(8)] + [
            i for i in range(9, n_layers, 2)
        ]
        # important that n_layers is even (32) here...
    elif cache_type == "half_no_first_8":
        keep_hidden_layer_idx = [i for i in range(1, n_layers, 2)]
    elif cache_type == "3":
        keep_hidden_layer_idx = [_ for _ in range(8)] + [
            i for i in range(10, n_layers, 3)
        ]
    elif cache_type == "3_no_first_8":
        keep_hidden_layer_idx = [i for i in range(1, n_layers, 3)]
    elif cache_type == "3_no_first_8_start_4":
        keep_hidden_layer_idx = [i for i in range(4, n_layers, 3)]
    elif cache_type == "4":
        keep_hidden_layer_idx = [_ for _ in range(8)] + [
            i for i in range(11, n_layers, 4)
        ]
    elif cache_type == "4_no_first_8":
        keep_hidden_layer_idx = [i for i in range(3, n_layers, 4)]
    elif cache_type == "4_no_first_8_start_1":
        keep_hidden_layer_idx = [i for i in range(1, n_layers, 4)]
    elif cache_type == "6":
        keep_hidden_layer_idx = [_ for _ in range(8)] + [
            i for i in range(13, n_layers, 6)
        ]
    elif cache_type == "10":
        keep_hidden_layer_idx = [_ for _ in range(8)] + [
            i for i in range(11, n_layers, 10)
        ]
    elif cache_type == "32":
        keep_hidden_layer_idx = [_ for _ in range(8)] + [
            i for i in range(31, n_layers, 32)
        ]
    elif cache_type == "degrade_ablation":
        if cache_args is None or cache_args > 3 or cache_args < 0:
            raise ValueError(f"got invalid args {cache_args} for {cache_type}")
        keep_hidden_layer_idx = [i for i in range(0, 8 * cache_args)] + [
            i for i in range(8 * (cache_args + 1), n_layers)
        ]

    elif cache_type == "recent":
        keep_hidden_layer_idx = [i for i in range(n_layers)]
        # self.cache_size = self.recent_size
        # to get a baseline?
    else:
        raise ValueError(f"got invalid cache_type {cache_type}")

    if n_layers - 1 not in keep_hidden_layer_idx:
        keep_hidden_layer_idx.append(n_layers - 1)
        ## Important???

    return keep_hidden_layer_idx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="models/mpt-7b-hscache"
    )
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="wikitext")

    parser.add_argument("--task", type=str, default="wikitext-2-raw-v1")
    parser.add_argument(
        "--split", type=str, default="test", choices=["validation", "test"]
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/debug",
    )

    parser.add_argument("--enable_start_recent_kv_cache", action="store_true")
    parser.add_argument("--enable_hscache", action="store_true")
    parser.add_argument("--enable_start_full_kv_cache", action="store_true")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=20)
    parser.add_argument("--cache_size", type=int, default=252)
    parser.add_argument("--enable_pos_shift", action="store_true")

    parser.add_argument("--num_eval_tokens", type=int, default=None)

    args = parser.parse_args()
    return args


def load(model_name_or_path):
    print(f"Loading model from {model_name_or_path} ...")
    # however, tensor parallel for running falcon will occur bugs
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    return model, tokenizer


def load_with_callback(model_name_or_path, callback):
    print(f"Loading model from {model_name_or_path} ...")
    # however, tensor parallel for running falcon will occur bugs
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model_config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model_config.device_map = "auto"
    model_config.torch_dtype = torch.float16
    model_config.trust_remote_code = True
    callback(model_config)
    model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    return model, tokenizer


def download_url(url: str, folder="folder"):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition("/")[2]
    file = file if file[0] == "?" else file.split("?")[0]
    path = osp.join(folder, file)
    if osp.exists(path):
        print(f"File {file} exists, use existing file.")
        return path

    print(f"Downloading {url}")
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, "wb") as f:
        f.write(data.read())

    return path


def load_jsonl(
    file_path,
):
    list_data_dict = []
    with open(file_path, "r") as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    return list_data_dict
