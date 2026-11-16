"""Trace tokenizer and complete decoder/encoder backbones."""
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F

def _validate_architecture(config):
    if (config.model_type != 'qwen3' or config.num_hidden_layers != 28
            or config.hidden_size != 2048):
        raise ValueError('Expected pretrained Qwen3-1.7B: qwen3, 28 layers, hidden_size=2048')


class MLPQwen3SCA(nn.Module):
    """Train the trace frontend, complete first decoder block, and AES head."""

    def __init__(self, backbone, trace_length, tokens=8, width=200):
        super().__init__()
        if backbone.config.model_type != 'qwen3' or len(backbone.layers) != 28:
            raise ValueError('MLPQwen3SCA requires a full 28-layer Qwen3Model')
        if min(trace_length, tokens, width) <= 0:
            raise ValueError('trace_length, tokens, and width must be positive')
        self.trace_length = trace_length
        self.tokens = tokens
        self.backbone = backbone.float()
        hidden_size = backbone.config.hidden_size
        self.frontend = nn.Sequential(
            nn.Linear(trace_length, width), nn.GELU(),
            nn.Linear(width, tokens * hidden_size),
        ).float()
        self.head = nn.Linear(hidden_size, 256).float()
        self.register_buffer('source_embed_scale', backbone.embed_tokens.weight.detach().std())
        self.backbone.requires_grad_(False)
        self.backbone.layers[0].requires_grad_(True)

    def forward(self, traces):
        if traces.ndim != 2 or traces.shape[1] != self.trace_length:
            raise ValueError(f'Expected traces with shape [batch, {self.trace_length}]')
        embedded = self.frontend(traces).reshape(traces.shape[0], self.tokens, -1)
        embedded = F.layer_norm(embedded, (self.backbone.config.hidden_size,)) * self.source_embed_scale
        hidden = self.backbone(inputs_embeds=embedded, use_cache=False).last_hidden_state
        return self.head(hidden.mean(dim=1))

    @property
    def trainable_names(self):
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    @property
    def parameter_counts(self):
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {'total': total, 'trainable': trainable, 'frozen': total - trainable}

    def optimizer_groups(self, io_lr, backbone_lr):
        io = [parameter for module in (self.frontend, self.head) for parameter in module.parameters()]
        backbone = list(self.backbone.layers[0].parameters())
        return [dict(name='io', params=io, lr=io_lr),
                dict(name='backbone', params=backbone, lr=backbone_lr)]


def build_mlp_qwen3(model_path, seed, trace_length, tokens=8):
    """Load complete local Qwen3-1.7B safetensors with FP32 master weights."""
    from transformers import AutoConfig, AutoModel
    model_path = Path(model_path)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    _validate_architecture(config)
    torch.manual_seed(seed)
    backbone, loading = AutoModel.from_pretrained(
        model_path, local_files_only=True, use_safetensors=True, dtype=torch.float32,
        attn_implementation='sdpa', output_loading_info=True,
    )
    _validate_architecture(backbone.config)
    bad_unexpected = [key for key in loading.get('unexpected_keys', []) if key != 'lm_head.weight']
    if (loading.get('missing_keys') or bad_unexpected or loading.get('mismatched_keys')
            or loading.get('error_msgs')):
        raise ValueError(f'Pretrained backbone loading was incomplete: {loading}')
    if any(parameter.dtype != torch.float32 for parameter in backbone.parameters()):
        raise ValueError('All pretrained master parameters must be float32')
    model = MLPQwen3SCA(backbone, trace_length=trace_length, tokens=tokens)
    model.loading_info = loading
    return model


class TraceLLM(MLPQwen3SCA):
    """Keep the original trace interface and source training scope, without a Qwen3-only gate."""

    def __init__(self, backbone, trace_length=500, tokens=8, width=200):
        nn.Module.__init__(self)
        if (backbone.config.model_type not in {'qwen2', 'llama'}
                or len(backbone.layers) != backbone.config.num_hidden_layers):
            raise ValueError('Require a complete supported decoder, without layer truncation')
        if min(trace_length, tokens, width) < 1:
            raise ValueError('Trace/interface dimensions must be positive')
        self.trace_length, self.tokens = trace_length, tokens
        self.backbone = backbone.float()
        hidden = backbone.config.hidden_size
        self.frontend = nn.Sequential(nn.Linear(trace_length, width), nn.GELU(),
                                      nn.Linear(width, tokens * hidden)).float()
        self.head = nn.Linear(hidden, 256).float()
        self.register_buffer('source_embed_scale', backbone.embed_tokens.weight.detach().std())
        self.backbone.requires_grad_(False)
        self.backbone.layers[0].requires_grad_(True)


class SelfAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = [layer(x).reshape(b, t, self.heads, d // self.heads).transpose(1, 2)
                   for layer in [self.q_proj, self.k_proj, self.v_proj]]
        h = F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=False)
        return self.out_proj(h.transpose(1, 2).reshape(b, t, d))


class EncoderBlock(nn.Module):
    def __init__(self, width, heads, ff):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.self_attn = SelfAttention(width, heads)
        self.linear1, self.linear2 = nn.Linear(width, ff), nn.Linear(ff, width)

    def forward(self, x):
        x = x + self.self_attn(self.norm1(x))
        return x + self.linear2(F.gelu(self.linear1(self.norm2(x))))


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=config['hidden_size'], num_hidden_layers=config['layers'])
        self.position = nn.Parameter(torch.empty(1, config['tokens'], config['hidden_size']))
        nn.init.normal_(self.position, std=.02)
        self.layers = nn.ModuleList([EncoderBlock(config['hidden_size'], config['heads'], config['feedforward_size'])
                                     for _ in range(config['layers'])])
        self.norm = nn.LayerNorm(config['hidden_size'])

    def forward(self, x):
        x = x + self.position
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class OrdinaryTransformer(nn.Module):
    def __init__(self, seed, trace_length=700, tokens=8, hidden_size=2048,
                 layers=8, heads=16, feedforward_size=8192):
        super().__init__()
        torch.manual_seed(seed)
        self.trace_length, self.tokens, self.hidden_size = trace_length, tokens, hidden_size
        config = dict(hidden_size=hidden_size, layers=layers, heads=heads,
                      feedforward_size=feedforward_size, tokens=tokens)
        self.frontend = nn.Sequential(nn.Linear(trace_length, 200), nn.GELU(),
                                      nn.Linear(200, tokens * hidden_size))
        self.token_norm = nn.LayerNorm(hidden_size)
        self.backbone = Encoder(config)
        self.head = nn.Linear(hidden_size, 256)

    def forward(self, traces):
        tokens = self.token_norm(self.frontend(traces).reshape(len(traces), self.tokens, self.hidden_size))
        return self.head(self.backbone(tokens).mean(dim=1))

    def optimizer_groups(self, io_lr=1e-3, backbone_lr=1e-4):
        decoder_ids = {id(p) for p in self.backbone.parameters()}
        return [dict(params=list(self.backbone.parameters()), lr=backbone_lr),
                dict(params=[p for p in self.parameters() if id(p) not in decoder_ids], lr=io_lr)]


class WideMLP(nn.Module):
    def __init__(self, seed, trace_length=700, width=2048, blocks=8):
        super().__init__()
        torch.manual_seed(seed)
        self.width, self.blocks = width, blocks
        self.frontend = nn.Sequential(nn.Linear(trace_length, 200), nn.GELU(), nn.Linear(200, width))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(blocks))
        self.fc1 = nn.ModuleList(nn.Linear(width, width) for _ in range(blocks))
        self.fc2 = nn.ModuleList(nn.Linear(width, width) for _ in range(blocks))
        self.final = nn.LayerNorm(width)
        self.head = nn.Linear(width, 256)

    def forward(self, x):
        h = self.frontend(x)
        for norm, a, b in zip(self.norms, self.fc1, self.fc2):
            h = h + b(F.gelu(a(norm(h))))
        return self.head(self.final(h))

    def optimizer_groups(self, io_lr=1e-3, backbone_lr=None):
        return [dict(params=list(self.parameters()), lr=io_lr)]


def build_model(config, seed, model_path=None):
    kind = config.get('backbone', 'qwen3')
    length, tokens = config.get('trace_length', 700), config.get('tokens', 8)
    if kind == 'transformer':
        return OrdinaryTransformer(seed, length, tokens, **config.get('encoder', {}))
    if kind == 'mlp':
        return WideMLP(seed, length, **config.get('mlp', {}))
    if model_path is None:
        raise ValueError('A local pretrained model directory is required for LLM backbones')
    if kind == 'qwen3':
        return build_mlp_qwen3(model_path, seed, length, tokens)
    if kind not in ('qwen2', 'minicpm'):
        raise ValueError('Unsupported backbone: ' + kind)
    from transformers import AutoModel
    torch.manual_seed(seed)
    backbone, info = AutoModel.from_pretrained(
        model_path, local_files_only=True, use_safetensors=True, dtype=torch.float32,
        attn_implementation='sdpa', output_loading_info=True)
    expected = {'qwen2': ('qwen2', 896, 24), 'minicpm': ('llama', 1536, 24)}[kind]
    actual = (backbone.config.model_type, backbone.config.hidden_size, len(backbone.layers))
    if actual != expected:
        raise ValueError(f'Expected complete backbone {expected}, received {actual}')
    unexpected = [key for key in info.get('unexpected_keys', []) if key != 'lm_head.weight']
    if unexpected or any(info.get(k) for k in ('missing_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('Pretrained backbone loading was incomplete')
    return TraceLLM(backbone, trace_length=length, tokens=tokens)
