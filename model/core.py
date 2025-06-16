# model/core.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

# Assuming these are available from your project structure
import model.hparams
import norm
import mask # For AlibiMask

# --- Sublayer Components (simplified for general purpose) ---

class TimeLerp(nn.Module):
    """
    A placeholder for a time mixing mechanism, potentially inspired by RWKV.
    In a real RWKV, this would involve complex state management and recurrence.
    For a general Transformer, this might not be needed or would be a simple Identity.
    Given its use with RWKVFeedForwardSubLayer, it suggests a non-standard attention.
    For this general purpose, it's a simple linear layer.
    """
    def __init__(self, dim: int):
        super().__init__()
        # In RWKV, this is more complex. Here, it's just a simple mixer.
        self.time_mix_x = nn.Parameter(torch.ones(1))
        self.time_mix_r = nn.Parameter(torch.ones(1))
        self.time_mix_g = nn.Parameter(torch.ones(1))

    def forward(self, x, state=None):
        # This is a very simplified placeholder.
        # Actual RWKV time mixing involves previous state and complex operations.
        # For now, it will just pass x through, assuming the attention handles temporal context.
        # If state is passed, it should be updated. For decoder-only, state is usually for inference.
        return x

class TorchAttention(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 n_head: int, 
                 d_qk: int, 
                 d_v: int, 
                 bias_mask_factory: Callable = None, 
                 dropout: float = 0.):
        super().__init__()
        self.n_head = n_head
        self.d_qk = d_qk
        self.d_v = d_v
        self.scale = d_qk ** -0.5

        self.q_proj = nn.Linear(d_model, n_head * d_qk, bias=False)
        self.k_proj = nn.Linear(d_model, n_head * d_qk, bias=False)
        self.v_proj = nn.Linear(d_model, n_head * d_v, bias=False)
        self.out_proj = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.bias_mask_factory = bias_mask_factory
        self.bias_mask = None # Will be initialized by factory if provided

        if self.bias_mask_factory:
            # Need to pass relevant args to factory, e.g., n_head, max_sequence_length
            # For now, we'll assume max_sequence_length is handled externally or factory needs no args
            self.bias_mask = self.bias_mask_factory(n_head=n_head, max_sequence_length=2048) # Placeholder max_seq_len

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.d_qk).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_head, self.d_qk).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_head, self.d_v).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply bias mask (e.g., Alibi or causal mask)
        if self.bias_mask:
            # Alibi mask is typically (n_head, seq_len, seq_len)
            # It gets added to attn_scores which is (batch_size, n_head, seq_len, seq_len)
            alibi_bias = self.bias_mask(seq_len, seq_len, x.device) # Assume mask needs current seq_len
            attn_scores = attn_scores + alibi_bias.unsqueeze(0) # Add batch dimension

        # Apply padding mask if provided
        if attention_mask is not None:
            # attention_mask should be (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, seq_len)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_head * self.d_v)
        output = self.out_proj(output)
        
        return output


class AttentionSubLayer(nn.Module):
    def __init__(self, 
                 d_model: int,
                 n_head: int,
                 d_qk: int,
                 d_v: int,
                 attention_factory: Callable,
                 qkv_norm_factory: Callable,
                 time_mixer_factory: Callable):
        super().__init__()
        self.qkv_norm = qkv_norm_factory(d_model)
        self.attention = attention_factory(
            d_model=d_model, n_head=n_head, d_qk=d_qk, d_v=d_v
        )
        self.time_mixer = time_mixer_factory(d_model)

    # Change this forward method:
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None): # <<< ADD attention_mask here
        normed_x = self.qkv_norm(x)
        mixed_x = self.time_mixer(normed_x) 
        return self.attention(mixed_x, attention_mask=attention_mask) # <<< PASS it to self.attention


class RWKVFeedForwardSubLayer(nn.Module):
    """
    A very simplified placeholder for an RWKV-style feedforward network.
    Actual RWKV FF involves a "channel mixing" component that also interacts
    with previous states and uses specific gating mechanisms.
    This general purpose version is just a standard FFN with a gating activation.
    """
    def __init__(self, d_model: int, d_feedforward: int, dropout: float = 0.):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_feedforward, bias=False)
        self.up_proj = nn.Linear(d_model, d_feedforward, bias=False)
        self.down_proj = nn.Linear(d_feedforward, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # A simple gated mechanism
        gated_output = F.silu(self.gate_proj(x)) * self.up_proj(x)
        output = self.down_proj(gated_output)
        return self.dropout(output)


class ResidualMixOp(nn.Module):
    # Line 149: Make sure this 'def' starts with exactly 4 spaces from the 'class' line
    def __init__(self, dim: int, sublayer_norm_factory: Callable):

        # These lines should be indented by 8 spaces from the 'class' line
        super().__init__() 
        self.norm = sublayer_norm_factory(dim) 

    # Make sure this 'def' starts with exactly 4 spaces from the 'class' line
    def forward(self, x: torch.Tensor, sublayer_fn: Callable, *args, **kwargs):
        # These lines should be indented by 8 spaces from the 'class' line
        normed_x = self.norm(x)
        sublayer_output = sublayer_fn(normed_x, *args, **kwargs)
        return x + sublayer_output

class TransformerLayer(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 hparams: model.hparams.HParams, # Pass hparams to get all needed dims
                 self_attention_sublayer_factory: Callable,
                 feedforward_sublayer_factory: Callable,
                 residual_op_factory: Callable):
        super().__init__()
        self.attn_norm = norm.RMSNorm(d_model, weight_scaling=False) # Pre-norm style
        self.ffn_norm = norm.RMSNorm(d_model, weight_scaling=False) # Pre-norm style

        # Attention sublayer
        self.self_attention_sublayer = self_attention_sublayer_factory(
            d_model=d_model,
            n_head=hparams.n_head,
            d_qk=hparams.d_qk,
            d_v=hparams.d_v,
        )
        
        # Feed-forward sublayer
        self.feedforward_sublayer = feedforward_sublayer_factory(
            d_model=d_model,
            d_feedforward=hparams.d_feedforward,
            dropout=hparams.dropout, # Assuming dropout is applicable here
        )
        
        # Custom residual operations
        # These factories are called *per sublayer* in the config.
        # This means each sublayer gets its own residual op instance.
        self.attn_residual = residual_op_factory()
        self.ffn_residual = residual_op_factory()

        # The `residual_op_factory` from the config gives `model.core.ResidualMixOp(sublayer_norm_factory=...)`
        # Let's adjust ResidualMixOp to be a simple pre-norm + residual add.
        # If the norm is already applied by the sublayer, then it's just `x + sublayer_output`.
        # Given the config, it specifies `sublayer_norm_factory` *inside* the residual op.
        # This suggests the residual op performs the normalization of `x` before passing to sublayer,
        # and then adds the residual.

        # Redefining ResidualMixOp for a standard pre-norm pattern used in Llama/GPT-NeoX/RWKV
        # Input `x` is normalized *before* being fed to the sublayer.
        # Then the sublayer output is added to the original `x`.
        class _ResidualMixOp(nn.Module):
            def __init__(self, sublayer_norm_factory: Callable, dim: int):
                super().__init__()
                self.norm = sublayer_norm_factory(dim)

            def forward(self, x: torch.Tensor, sublayer: nn.Module):
                # Apply norm to x before feeding to sublayer
                normed_x = self.norm(x)
                sublayer_output = sublayer(normed_x)
                return x + sublayer_output

        self.attn_residual_op = residual_op_factory()
        self.ffn_residual_op = residual_op_factory()

        # To avoid circular import for HParams in the factory calls within main Decoder
        # The hparams object is passed to TransformerLayer
        
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        # Attention sublayer with residual connection
        x = self.attn_residual_op(x, lambda input_x: self.self_attention_sublayer(input_x, attention_mask))
        
        # Feed-forward sublayer with residual connection
        x = self.ffn_residual_op(x, self.feedforward_sublayer)
        
        return x

class Decoder(nn.Module):
    def __init__(self, 
                 hparams: model.hparams.HParams,
                 embedding_norm_factory: Callable,
                 positional_embedding_factory: Callable,
                 share_embedding_weights: bool,
                 layer_factory: Callable,
                 final_norm_factory: Callable):
        super().__init__()
        self.hparams = hparams

        self.token_embeddings = nn.Embedding(hparams.vocab_size, hparams.d_model)
        
        self.embedding_norm = embedding_norm_factory(hparams.d_model)
        self.positional_embedding = positional_embedding_factory()

        self.layers = nn.ModuleList([
            layer_factory(
                d_model=hparams.d_model,
                hparams=hparams, # Pass hparams to each layer
            )
            for _ in range(hparams.n_layer)
        ])

        self.final_norm = final_norm_factory(hparams.d_model)
        self.lm_head = nn.Linear(hparams.d_model, hparams.vocab_size, bias=False)

        if share_embedding_weights:
            self.lm_head.weight = self.token_embeddings.weight

        # Initialize weights (often done here, or using specific init functions)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Normal initialization for linear layers
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Normal initialization for embeddings
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, norm.RMSNorm):
            if hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter):
                nn.init.ones_(module.weight) # RMSNorm weights are usually initialized to 1

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        # input_ids: (batch_size, sequence_length)
        # attention_mask: (batch_size, sequence_length) - 1 for real tokens, 0 for padding

        x = self.token_embeddings(input_ids)
        x = self.embedding_norm(x)
        
        # Positional embedding (identity in this config)
        x = self.positional_embedding(x)

        # Create a causal attention mask (lower triangular) if not already provided or if padding mask is separate
        # A simple causal mask (for decoder) + padding mask (if attention_mask is given as padding)
        seq_len = input_ids.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1).bool()
        
        if attention_mask is not None:
            # Expand attention_mask to be compatible with multi-head attention (b, 1, 1, s)
            # and combine with causal mask
            padding_mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2).bool() # (batch, 1, 1, seq_len)
            # Combine causal mask with padding mask
            # True for valid positions, False for masked. Invert for masked_fill
            combined_mask = ~(padding_mask_expanded & ~causal_mask) # True where should be masked (-inf)
            final_attn_mask = combined_mask.unsqueeze(1) # (b, 1, seq_len, seq_len)
        else:
            final_attn_mask = causal_mask.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len)

        for layer in self.layers:
            x = layer(x, attention_mask=final_attn_mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits
