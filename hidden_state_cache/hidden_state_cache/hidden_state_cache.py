import torch
from typing import Any, Dict, Optional, Tuple, Union
from .low_rank_qkv import (
    LowRankQKV,
    FullRankQKV,
)
from hidden_state_cache.utils import get_cache_type

# TODO: deprecate LowRankQKVList and LowRankQKV


def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


class HiddenStateCache:
    def __init__(
        self,
        batch_size,
        n_layers,
        keep_hidden_layer_idx=None,
        cache_type="full",
        cache_args=None,
        seq_dim=1,
        start_size=4,  # D: number of sink tokens
        recent_size=50,
        cache_size=2048,  # just as much as need lol
        baseline=False,
        # kv_transform: Optional[Union[LowRankQKVList, LowRankQKVEncoderList]] = None,
        low_rank_qkv: Optional[LowRankQKV] = None,
    ):
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.start_size = start_size
        self.cache_size = cache_size
        self.recent_size = recent_size
        self.seq_dim = seq_dim
        self.seq_slice = DIM_TO_SLICE[seq_dim]
        self.baseline = baseline
        # self.kv_transform = kv_transform
        self.low_rank_qkv = low_rank_qkv

        self.hidden_states = None

        if keep_hidden_layer_idx is None:
            self.keep_hidden_layer_idx = get_cache_type(
                cache_type, self.n_layers, cache_args
            )
            if cache_type == "recent":
                self.recent_size = self.cache_size
                # to get a baseline?

        else:
            self.keep_hidden_layer_idx = sorted(keep_hidden_layer_idx)

        self.layer_replace_idx = []
        j = 0
        for i in self.keep_hidden_layer_idx:
            while i >= j:
                self.layer_replace_idx.append(i)
                j += 1

    def slice_hidden_states(self, layer_hidden_state, layer_idx):
        if layer_idx in self.keep_hidden_layer_idx:
            max_len = self.cache_size
        else:
            max_len = self.recent_size

        if layer_hidden_state.size(self.seq_dim) <= max_len + self.start_size:
            # bug: forgot to add self.start_size here
            return layer_hidden_state
        else:
            return torch.cat(
                [
                    self.seq_slice(layer_hidden_state, 0, self.start_size),
                    self.seq_slice(
                        layer_hidden_state,
                        layer_hidden_state.size(self.seq_dim) - max_len,
                        layer_hidden_state.size(self.seq_dim),
                    ),
                ],
                dim=self.seq_dim,
            )

    def add_to_cache(self, incoming_hidden_states):
        """
        We add hidden_states to the cache, and evict if we go over self.cache_size

        incoming_hidden_states: tuple[torch.Tensor], n_layers+1 * [batch_size, seq_length, hidden_size]
        """

        # if isinstance(self.kv_transform, LowRankQKVEncoderList):
        #     incoming_hidden_states = [
        #         self.kv_transform.encode_kv(layer_idx, incoming_hidden_layer)
        #         for layer_idx, incoming_hidden_layer in enumerate(
        #             incoming_hidden_states[: self.n_layers]
        #         )
        #     ]

        if self.low_rank_qkv is not None:
            incoming_hidden_states = [
                self.low_rank_qkv.encode(layer_idx, incoming_hidden_layer)
                for layer_idx, incoming_hidden_layer in enumerate(
                    incoming_hidden_states[: self.n_layers]
                )
            ]

        if not self.hidden_states:
            self.hidden_states = [
                self.slice_hidden_states(incoming_hidden_layer, layer_idx)
                for layer_idx, incoming_hidden_layer in enumerate(
                    incoming_hidden_states[: self.n_layers]
                )
            ]

        else:
            self.hidden_states = [
                self.slice_hidden_states(
                    torch.cat(
                        [
                            past_hidden_layer,
                            incoming_hidden_layer,
                            ### Bug: I had the incoming_hidden_layer before past_hidden_layer!!!
                        ],
                        dim=self.seq_dim,
                    ),
                    layer_idx,
                )
                for layer_idx, (incoming_hidden_layer, past_hidden_layer) in enumerate(
                    zip(incoming_hidden_states[: self.n_layers], self.hidden_states)
                )
            ]

    def add_to_cache_fast(self, incoming_hidden_states):
        if self.low_rank_qkv is not None:
            incoming_hidden_states = [
                self.low_rank_qkv.encode(layer_idx, incoming_hidden_layer)
                for layer_idx, incoming_hidden_layer in enumerate(
                    incoming_hidden_states[: self.n_layers]
                )
            ]

        if not self.hidden_states:
            self.hidden_states = incoming_hidden_states[: self.n_layers]

        else:
            self.hidden_states = [
                torch.cat(
                    [
                        past_hidden_layer,
                        incoming_hidden_layer,
                        ### Bug: I had the incoming_hidden_layer before past_hidden_layer!!!
                    ],
                    dim=self.seq_dim,
                )
                for layer_idx, (incoming_hidden_layer, past_hidden_layer) in enumerate(
                    zip(incoming_hidden_states[: self.n_layers], self.hidden_states)
                )
            ]

    def get_layer(self, layer_idx):
        ### Shouldn't call get_layer from outside if we are using the encoder
        ### This is why supporting both is kinda messy code lmao...
        ### doing this because I will eventually delete the older version...

        if self.hidden_states is None:
            return None  # empty cache... Be careful when handling??

        cache_seq_len = self.hidden_states[self.layer_replace_idx[0]].size(self.seq_dim)
        # max number in cache already
        if (
            layer_idx in self.keep_hidden_layer_idx
            or cache_seq_len <= self.recent_size + self.start_size
            or self.baseline  ### we don't want to replace evicted hidden states!!
        ):
            return self.hidden_states[layer_idx]
        else:
            assert (
                self.hidden_states[layer_idx].size(self.seq_dim)
                == self.recent_size + self.start_size
            )
            return torch.cat(
                [
                    self.seq_slice(
                        self.hidden_states[layer_idx],
                        0,
                        self.start_size,
                    ),
                    self.seq_slice(
                        self.hidden_states[self.layer_replace_idx[layer_idx]],
                        self.start_size,
                        cache_seq_len - self.recent_size,
                    ),
                    self.seq_slice(
                        self.hidden_states[layer_idx],
                        self.start_size,
                        self.recent_size + self.start_size,
                    ),  # forgot the + self.start_size here???
                ],
                dim=self.seq_dim,
            )
