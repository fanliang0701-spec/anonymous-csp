from types import SimpleNamespace
import pytest
import torch
from torch import nn
from csp_sca.models import MLPQwen3SCA, _validate_architecture


class SmallBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type='qwen3', hidden_size=8, num_hidden_layers=28)
        self.embed_tokens = nn.Embedding(32, 8)
        self.layers = nn.ModuleList(nn.Linear(8, 8) for _ in range(28))

    def forward(self, inputs_embeds, use_cache):
        assert use_cache is False
        self.input_seen = inputs_embeds.detach()
        x = inputs_embeds
        for layer in self.layers:
            x = x + .01 * layer(x)
        return SimpleNamespace(last_hidden_state=x)


def test_trace_tokenizer_scale_shape_and_source_training_scope():
    model = MLPQwen3SCA(SmallBackbone(), trace_length=12, tokens=4)
    output = model(torch.randn(3, 12))
    assert output.shape == (3, 256)
    assert model.backbone.input_seen.shape == (3, 4, 8)
    torch.testing.assert_close(model.backbone.input_seen.mean(-1), torch.zeros(3, 4), atol=1e-6, rtol=0)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert all(name.startswith(('frontend.', 'head.', 'backbone.layers.0.')) for name in trainable)
    assert not model.backbone.embed_tokens.weight.requires_grad
    groups = model.optimizer_groups(1e-3, 1e-5)
    assert {id(p) for g in groups for p in g['params']} == {id(p) for p in model.parameters() if p.requires_grad}
    with pytest.raises(ValueError, match='Qwen3-1.7B'):
        _validate_architecture(model.backbone.config)
