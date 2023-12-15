"""GPT Blocks used for the GPT Model."""
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from .attention import ATTN_CLASS_REGISTRY
from .ffn import FFN_CLASS_REGISTRY, build_ffn
from .norm import NORM_CLASS_REGISTRY
from hidden_state_cache.hidden_state_cache.hidden_state_cache import HiddenStateCache


class MPTBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        expansion_ratio: int,
        attn_config: Optional[Dict] = None,
        ffn_config: Optional[Dict] = None,
        resid_pdrop: float = 0.0,
        norm_type: str = "low_precision_layernorm",
        fc_type: str = "torch",
        device: Optional[str] = None,
        no_bias: bool = False,
        **kwargs: Any
    ):
        if attn_config is None:
            attn_config = {
                "attn_type": "multihead_attention",
                "attn_pdrop": 0.0,
                "attn_impl": "triton",
                "qk_ln": False,
                "clip_qkv": None,
                "softmax_scale": None,
                "prefix_lm": False,
                "attn_uses_sequence_id": False,
                "alibi": False,
                "alibi_bias_max": 8,
            }
        if ffn_config is None:
            ffn_config = {"ffn_type": "mptmlp"}
        del kwargs
        super().__init__()
        norm_class = NORM_CLASS_REGISTRY[norm_type.lower()]
        assert isinstance(attn_config["attn_type"], str)
        attn_class = ATTN_CLASS_REGISTRY[attn_config["attn_type"]]
        args_to_exclude_in_attn_class = {
            "attn_type",
            "prefix_lm",
            "alibi",
            "attn_uses_sequence_id",
            "alibi_bias_max",
        }
        attn_config_subset_for_attn_class = {
            k: v
            for (k, v) in attn_config.items()
            if k not in args_to_exclude_in_attn_class
        }
        self.norm_1 = norm_class(d_model, device=device)
        self.attn = attn_class(
            d_model=d_model,
            n_heads=n_heads,
            fc_type=fc_type,
            device=device,
            **attn_config_subset_for_attn_class,
            bias=not no_bias
        )
        self.norm_2 = None
        if not getattr(FFN_CLASS_REGISTRY[ffn_config["ffn_type"]], "_has_norm", False):
            self.norm_2 = norm_class(d_model, device=device)
        self.ffn = build_ffn(
            d_model=d_model,
            expansion_ratio=expansion_ratio,
            device=device,
            bias=not no_bias,
            **ffn_config
        )
        self.resid_attn_dropout = nn.Dropout(resid_pdrop)
        self.resid_ffn_dropout = nn.Dropout(resid_pdrop)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        # past_hidden_state: Optional[torch.Tensor] = None,
        hidden_state_cache: Optional[HiddenStateCache] = None,
        layer_idx: Optional[int] = None,
        attn_bias: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        is_causal: bool = True,
        output_attentions: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        a = self.norm_1(x)
        # if hidden_state_cache is not None:
        #     past_hidden_state = self.norm_1(past_hidden_state)
        (b, attn_weights, past_key_value) = self.attn(
            a,
            past_key_value=past_key_value,
            # past_hidden_state=past_hidden_state,
            hidden_state_cache=hidden_state_cache,
            layer_idx=layer_idx,
            attn_bias=attn_bias,
            attention_mask=attention_mask,
            is_causal=is_causal,
            needs_weights=output_attentions,
        )
        x = x + self.resid_attn_dropout(b)
        m = x
        if self.norm_2 is not None:
            m = self.norm_2(x)
        n = self.ffn(m)
        x = x + self.resid_ffn_dropout(n)
        return (x, attn_weights, past_key_value)
