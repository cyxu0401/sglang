# Copyright 2026 SGLang contributors
# SPDX-License-Identifier: Apache-2.0
"""MiniCPM5 gated-attention MoE with TP=EP=PP=1."""

from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.hf_transformers import get_rope_config
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_parallel


class MiniCPM5MLP(nn.Module):
    """Dense/shared SwiGLU with quantization and unchanged HF module names."""

    def __init__(self, config, intermediate_size: int, prefix: str, quant_config=None):
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError("MiniCPM5 currently requires hidden_act='silu'")
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_proj", prefix),
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )

    def forward(self, hidden_states):
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        output, _ = self.down_proj(F.silu(gate) * up)
        return output


class MiniCPM5Router(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size),
            requires_grad=False,
        )
        # The source buffer is FP32. A non-trainable parameter lets the standard
        # SGLang loader preserve this dtype while TopK keeps a reference to it.
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(config.n_routed_experts, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, hidden_states):
        return F.linear(hidden_states.float(), self.weight.float())


class MiniCPM5MoE(nn.Module):
    def __init__(self, config, layer_id, quant_config, prefix):
        super().__init__()
        self.gate = MiniCPM5Router(config)
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            use_grouped_topk=True,
            topk_group=config.topk_group,
            num_expert_group=config.n_group,
            renormalize=config.norm_topk_prob,
            scoring_func="sigmoid",
            correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=config.routed_scaling_factor,
            # Fold the routed scale into TopK weights exactly once. Shared
            # experts are evaluated separately and retain their unit weight.
            apply_routed_scaling_factor_on_output=True,
        )
        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            activation="silu",
            inplace=False,
            reduce_results=False,
            num_fused_shared_experts=0,
            routing_method_type=RoutingMethodType.DeepSeekV3,
        )
        self.shared_experts = (
            MiniCPM5MLP(
                config,
                config.moe_intermediate_size * config.n_shared_experts,
                add_prefix("shared_experts", prefix),
                quant_config=quant_config,
            )
            if config.n_shared_experts
            else None
        )

    def forward(self, hidden_states, forward_batch=None):
        if hidden_states.shape[0] == 0:
            return hidden_states
        shared = (
            self.shared_experts(hidden_states)
            if self.shared_experts is not None
            else None
        )
        router_logits = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        output = self.experts(hidden_states, topk_output)
        return output + shared if shared is not None else output

    def get_moe_weights(self):
        return [parameter.data for parameter in self.experts.parameters()]


class MiniCPM5Attention(nn.Module):
    def __init__(self, config, layer_id, quant_config, prefix):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.use_gated_attention = config.use_gated_attention
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        q_out = self.q_size * (2 if self.use_gated_attention else 1)
        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            q_out,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("q_proj", prefix),
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,
            self.kv_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("k_proj", prefix),
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size,
            self.kv_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("v_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.q_size,
            config.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        rope_theta, rope_scaling = get_rope_config(config)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=True,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(self, positions, hidden_states, forward_batch):
        query, _ = self.q_proj(hidden_states)
        key, _ = self.k_proj(hidden_states)
        value, _ = self.v_proj(hidden_states)
        gate = None
        if self.use_gated_attention:
            # Checkpoint rows are [q_head_0, gate_head_0, q_head_1, ...].
            query, gate = query.reshape(-1, self.num_heads, 2 * self.head_dim).chunk(
                2, dim=-1
            )
            query = query.reshape(-1, self.q_size).contiguous()
            gate = gate.reshape(-1, self.q_size).contiguous()
        query, key = self.rotary_emb(positions, query, key)
        output = self.attn(query, key, value, forward_batch)
        if gate is not None:
            output = output * torch.sigmoid(gate)
        output, _ = self.o_proj(output)
        return output


class MiniCPM5DecoderLayer(nn.Module):
    def __init__(self, config, layer_id, quant_config, prefix):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attn = MiniCPM5Attention(
            config,
            layer_id,
            quant_config,
            add_prefix("self_attn", prefix),
        )
        self.mlp = (
            MiniCPM5MLP(
                config,
                config.intermediate_size,
                add_prefix("mlp", prefix),
                quant_config=quant_config,
            )
            if layer_id < config.first_k_dense_replace
            else MiniCPM5MoE(config, layer_id, quant_config, add_prefix("mlp", prefix))
        )

    def forward(self, positions, hidden_states, forward_batch):
        residual = hidden_states
        hidden_states = self.self_attn(
            positions, self.input_layernorm(hidden_states), forward_batch
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class MiniCPM5Model(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.config = config
        self.dspark_capture_layers: tuple[int, ...] = ()
        self.start_layer = 0
        self.end_layer = config.num_hidden_layers
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(
            [
                MiniCPM5DecoderLayer(
                    config,
                    layer_id,
                    quant_config,
                    add_prefix(f"layers.{layer_id}", prefix),
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        captured = {}
        for layer_id, layer in enumerate(self.layers):
            hidden_states = layer(positions, hidden_states, forward_batch)
            if layer_id in self.dspark_capture_layers:
                # Decoder layers return a new post-residual tensor.
                captured[layer_id] = hidden_states
        hidden_states = self.norm(hidden_states)
        if self.dspark_capture_layers:
            return hidden_states, [captured[i] for i in self.dspark_capture_layers]
        return hidden_states


class MiniCPM5MoEForCausalLM(nn.Module):
    packed_modules_mapping = {}

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        parallel = get_parallel()
        if parallel.tp_size != 1 or parallel.moe_ep_size != 1 or parallel.pp_size != 1:
            raise NotImplementedError("MiniCPM5 MoE currently requires TP=EP=PP=1")
        self.config = config
        self.quant_config = quant_config
        self.model = MiniCPM5Model(config, quant_config, add_prefix("model", prefix))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=None,
            prefix=add_prefix("lm_head", prefix),
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self, input_ids, positions, forward_batch: ForwardBatch, input_embeds=None
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            aux_hidden_states=aux_hidden_states,
        )

    @property
    def capture_aux_hidden_states(self) -> bool:
        return bool(self.model.dspark_capture_layers)

    def set_dspark_layers_to_capture(self, layer_ids: list[int]) -> None:
        if not layer_ids or any(i < 0 or i >= self.model.end_layer for i in layer_ids):
            raise ValueError(f"Invalid MiniCPM5 DSpark target layer IDs: {layer_ids}")
        self.model.dspark_capture_layers = tuple(layer_ids)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=config.n_group,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        expert_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )
        parameters = dict(self.named_parameters())
        loaded = set()
        for source_name, weight in weights:
            if "rotary_emb." in source_name:
                continue
            if self.config.tie_word_embeddings and source_name == "lm_head.weight":
                continue
            if ".mlp.experts." in source_name:
                for param_name, weight_name, expert_id, shard_id in expert_mapping:
                    if weight_name in source_name:
                        name = source_name.replace(weight_name, param_name)
                        if name not in parameters:
                            raise KeyError(
                                f"Unmapped expert tensor {source_name} -> {name}"
                            )
                        parameter = parameters[name]
                        parameter.weight_loader(
                            parameter,
                            weight,
                            name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                        loaded.add(name)
                        break
                else:
                    raise KeyError(f"Unmapped expert tensor {source_name}")
            else:
                if source_name not in parameters:
                    raise KeyError(
                        f"Unexpected MiniCPM5 checkpoint tensor {source_name}"
                    )
                parameter = parameters[source_name]
                if source_name.endswith(".gate.e_score_correction_bias") and (
                    parameter.dtype != torch.float32 or weight.dtype != torch.float32
                ):
                    raise ValueError(
                        f"Router correction bias must use FP32: {source_name}"
                    )
                getattr(parameter, "weight_loader", default_weight_loader)(
                    parameter, weight
                )
                loaded.add(source_name)
        return loaded


EntryClass = MiniCPM5MoEForCausalLM
