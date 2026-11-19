import pytest
import torch
from torch import nn

from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
    configure_aux_hidden_state_capture,
)
from sglang.srt.models.minicpm5_moe import MiniCPM5Model, MiniCPM5MoEForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _AddResidual(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, positions, hidden_states, forward_batch):
        return hidden_states + self.value


def test_dspark_captures_post_residual_states_in_draft_order():
    # Substitute the decoder layers to exercise capture without a GPU kernel.
    model = MiniCPM5Model.__new__(MiniCPM5Model)
    nn.Module.__init__(model)
    model.start_layer = 0
    model.end_layer = 3
    model.dspark_capture_layers = ()
    model.layers = nn.ModuleList(_AddResidual(value) for value in (1, 2, 4))
    model.embed_tokens = nn.Embedding(4, 2)
    model.norm = nn.Identity()

    target = MiniCPM5MoEForCausalLM.__new__(MiniCPM5MoEForCausalLM)
    nn.Module.__init__(target)
    target.model = model
    target.lm_head = nn.Identity()

    def capture_logits(
        input_ids, hidden_states, lm_head, forward_batch, aux_hidden_states
    ):
        return hidden_states, aux_hidden_states

    target.logits_processor = capture_logits
    input_ids = torch.tensor([0, 1])
    positions = torch.arange(len(input_ids))
    embeds = torch.zeros((2, 2))

    full, aux = target(input_ids, positions, None, input_embeds=embeds)
    assert not target.capture_aux_hidden_states
    assert aux is None

    configure_aux_hidden_state_capture(
        model=target,
        eagle_use_aux_hidden_state=False,
        eagle_aux_hidden_state_layer_ids=None,
        dflash_use_aux_hidden_state=True,
        dflash_target_layer_ids=[2, 0],
        is_dspark=True,
    )
    assert target.capture_aux_hidden_states
    assert target.get_input_embeddings() is model.embed_tokens

    captured, aux = target(input_ids, positions, None, input_embeds=embeds)
    torch.testing.assert_close(captured, full)
    torch.testing.assert_close(aux[0], embeds + 7)
    torch.testing.assert_close(aux[1], embeds + 1)

    with pytest.raises(ValueError, match="Invalid MiniCPM5 DSpark target layer"):
        target.set_dspark_layers_to_capture([3])
