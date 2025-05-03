import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import torch.quantization
@dataclass
class SimpleConfig:
    """Simple configuration class with essential parameters"""
    vocab_size: int = 32000
    d_model: int = 768
    n_heads: int = 10
    n_layers: int = 12
    mlp_ratio: float = 4.0
    max_seq_length: int = 2048
    dropout: float = 0.1
    rope_theta: float = 10000.0
    
    @property
    def head_dim(self):
        return self.d_model // self.n_heads
    
    @property
    def hidden_size(self):
        return int(self.mlp_ratio * self.d_model)

class RMSLayerNorm(nn.Module):
    """RMS layer normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight

class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings"""
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        cos, sin = self._compute_cos_sin_cache(max_seq_len)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        
    def _compute_cos_sin_cache(self, seq_len):
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        seq = torch.arange(seq_len, dtype=torch.float)
        freqs = torch.outer(seq, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().view(1, 1, seq_len, self.dim)
        sin = emb.sin().view(1, 1, seq_len, self.dim)
        return cos, sin
        
    def forward(self, q, k, seq_len=None):
        if seq_len is None:
            seq_len = q.size(2)
        if seq_len > self.max_seq_len:
            cos, sin = self._compute_cos_sin_cache(seq_len)
            self.register_buffer("cos_cached", cos, persistent=False)
            self.register_buffer("sin_cached", sin, persistent=False)
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        q_rot = torch.cat([-q2, q1], dim=-1)
        k_rot = torch.cat([-k2, k1], dim=-1)
        
        q = q * cos + q_rot * sin
        k = k * cos + k_rot * sin
        return q, k

class AttentionBlock(nn.Module):
    """Self-attention block similar to LLaMA"""
    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.config = config
        
        self.norm = RMSLayerNorm(config.d_model)
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.rope = RotaryEmbedding(
            config.head_dim, 
            max_seq_len=config.max_seq_length,
            theta=config.rope_theta
        )
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        bsz, seqlen, _ = x.shape
        h = self.norm(x)
        q = self.q_proj(h).view(bsz, seqlen, self.config.n_heads, self.config.head_dim).transpose(1,2)
        k = self.k_proj(h).view(bsz, seqlen, self.config.n_heads, self.config.head_dim).transpose(1,2)
        v = self.v_proj(h).view(bsz, seqlen, self.config.n_heads, self.config.head_dim).transpose(1,2)
        
        q, k = self.rope(q, k, seqlen)
        scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.config.head_dim)
        mask = torch.triu(torch.ones(seqlen, seqlen, device=x.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), -float('inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1,2).contiguous().view(bsz, seqlen, self.config.d_model)
        out = self.o_proj(out)
        return x + self.dropout(out)

class FeedForwardBlock(nn.Module):
    """Feed-forward block with SwiGLU activation"""
    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.config = config
        
        self.norm = RMSLayerNorm(config.d_model)
        self.w1 = nn.Linear(config.d_model, config.hidden_size, bias=False)
        self.w2 = nn.Linear(config.d_model, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        h = self.norm(x)
        h1 = self.w1(h)
        h2 = self.w2(h)
        hidden = F.silu(h1) * h2
        out = self.w3(hidden)
        return x + self.dropout(out)

class TransformerBlock(nn.Module):
    """Combined attention+FFN, with optional FiLM conditioning"""
    def __init__(
        self,
        config: SimpleConfig,
        cond_dim: Optional[int] = None,
        cond_predict_scale: bool = False
    ):
        super().__init__()
        self.config = config
        self.attention = AttentionBlock(config)
        self.feed_forward = FeedForwardBlock(config)

        # if cond_dim is given, build a little FiLM MLP
        self.cond_dim = cond_dim
        self.cond_predict_scale = cond_predict_scale
        if cond_dim is not None:
            out_ch = config.d_model * (2 if cond_predict_scale else 1)
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, out_ch)
            )
        else:
            self.cond_encoder = None

    def forward(self, x, cond: Optional[torch.Tensor] = None):
        # --- self-attention + residual ---
        x = self.attention(x)

        # --- FiLM conditioning (if requested) ---
        if self.cond_encoder is not None:
            # cond: [batch, cond_dim]
            cond = cond.to(x.dtype)
            emb = self.cond_encoder(cond)  # [batch, out_ch]
            bsz = emb.size(0)
            if self.cond_predict_scale:
                # split to scale & bias
                emb = emb.view(bsz, 2, self.config.d_model)
                scale = emb[:,0].unsqueeze(1)   # [batch,1,d_model]
                bias  = emb[:,1].unsqueeze(1)
                x = scale * x + bias
            else:
                bias = emb.unsqueeze(1)        # [batch,1,d_model]
                x = x + bias

        # --- feed-forward + residual ---
        x = self.feed_forward(x)
        return x

class SimpleLLaDAModel(nn.Module):
    """Simplified LLaDA with FiLM conditioning"""
    def __init__(
        self,
        config: SimpleConfig,
        cond_dim: int,
        cond_predict_scale: bool = True
    ):
        super().__init__()
        self.config = config
        self.cond_dim = cond_dim
        self.cond_predict_scale = cond_predict_scale

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(config, cond_dim, cond_predict_scale)
            for _ in range(config.n_layers)
        ])
        self.norm = RMSLayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, cond: torch.Tensor):
        """
        input_ids: [batch, seq_len]
        cond:      [batch, cond_dim]
        """
        x = self.token_embedding(input_ids)   # [batch, seq_len, d_model]
        for block in self.blocks:
            x = block(x, cond)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


def create_model(vocab_size=32000, d_model=100, n_heads=10, n_layers=12):
    """Helper function to create a model with default parameters"""
    config = SimpleConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    )
    return SimpleLLaDAModel(config, 
                            cond_dim=529)
# Custom model



# vocab_size = 1024
# batch_size = 1
# seq_len = 29*16

# # Create random token indices (0 to vocab_size-1)
# input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
# cond = torch.rand((batch_size, 100))

# # Initialize model
# model = create_model(vocab_size=vocab_size)  # Uses default config

# # Forward pass
# with torch.no_grad():
#     logits = model(input_ids, cond)
    
# print("Output shape:", logits.shape)  # Should be (1, 5, 32000)
# print("Sample output:", logits[0, -1, :5])  # First 5 logits of last token

