# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
from typing import Optional

from MeshAnything.miche.michelangelo.models.modules.checkpoint import checkpoint
from MeshAnything.miche.michelangelo.models.modules.transformer_blocks import (
    init_linear,
    MLP,
    MultiheadCrossAttention,
    MultiheadAttention,
    ResidualAttentionBlock
)


class AdaLayerNorm(nn.Module):
    def __init__(self,
                 device: torch.device,
                 dtype: torch.dtype,
                 width: int):

        super().__init__()

        self.silu = nn.SiLU(inplace=True)
        self.linear = nn.Linear(width, width * 2, device=device, dtype=dtype)
        self.layernorm = nn.LayerNorm(width, elementwise_affine=False, device=device, dtype=dtype)

    def forward(self, x, timestep):
        emb = self.linear(timestep)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class DitBlock(nn.Module):
    def __init__(
            self,
            *,
            device: torch.device,
            dtype: torch.dtype,
            n_ctx: int,
            width: int,
            heads: int,
            context_dim: int,
            qkv_bias: bool = False,
            init_scale: float = 1.0,
            use_checkpoint: bool = False
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.attn = MultiheadAttention(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias
        )
        self.ln_1 = AdaLayerNorm(device, dtype, width)

        if context_dim is not None:
            self.ln_2 = AdaLayerNorm(device, dtype, width)
            self.cross_attn = MultiheadCrossAttention(
                device=device,
                dtype=dtype,
                width=width,
                heads=heads,
                data_width=context_dim,
                init_scale=init_scale,
                qkv_bias=qkv_bias
            )

        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        self.ln_3 = AdaLayerNorm(device, dtype, width)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: Optional[torch.Tensor] = None):
        return checkpoint(self._forward, (x, t, context), self.parameters(), self.use_checkpoint)

    def _forward(self, x: torch.Tensor, t: torch.Tensor, context: Optional[torch.Tensor] = None):
        x = x + self.attn(self.ln_1(x, t))
        if context is not None:
            x = x + self.cross_attn(self.ln_2(x, t), context)
        x = x + self.mlp(self.ln_3(x, t))
        return x


class DiT(nn.Module):
    def __init__(
            self,
            *,
            device: Optional[torch.device],
            dtype: Optional[torch.dtype],
            n_ctx: int,
            width: int,
            layers: int,
            heads: int,
            context_dim: int,
            init_scale: float = 0.25,
            qkv_bias: bool = False,
            use_checkpoint: bool = False
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers

        self.resblocks = nn.ModuleList(
            [
                DitBlock(
                    device=device,
                    dtype=dtype,
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    context_dim=context_dim,
                    qkv_bias=qkv_bias,
                    init_scale=init_scale,
                    use_checkpoint=use_checkpoint
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: Optional[torch.Tensor] = None):
        for block in self.resblocks:
            x = block(x, t, context)
        return x


class UNetDiffusionTransformer(nn.Module):
    def __init__(
            self,
            *,
            device: Optional[torch.device],
            dtype: Optional[torch.dtype],
            n_ctx: int,
            width: int,
            layers: int,
            heads: int,
            init_scale: float = 0.25,
            qkv_bias: bool = False,
            skip_ln: bool = False,
            use_checkpoint: bool = False
    ):
        super().__init__()

        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers

        self.encoder = nn.ModuleList()
        for _ in range(layers):
            resblock = ResidualAttentionBlock(
                device=device,
                dtype=dtype,
                n_ctx=n_ctx,
                width=width,
                heads=heads,
                init_scale=init_scale,
                qkv_bias=qkv_bias,
                use_checkpoint=use_checkpoint
            )
            self.encoder.append(resblock)

        self.middle_block = ResidualAttentionBlock(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_checkpoint=use_checkpoint
        )

        self.decoder = nn.ModuleList()
        for _ in range(layers):
            resblock = ResidualAttentionBlock(
                device=device,
                dtype=dtype,
                n_ctx=n_ctx,
                width=width,
                heads=heads,
                init_scale=init_scale,
                qkv_bias=qkv_bias,
                use_checkpoint=use_checkpoint
            )
            linear = nn.Linear(width * 2, width, device=device, dtype=dtype)
            init_linear(linear, init_scale)

            layer_norm = nn.LayerNorm(width, device=device, dtype=dtype) if skip_ln else None

            self.decoder.append(nn.ModuleList([resblock, linear, layer_norm]))

    def forward(self, x: torch.Tensor):

        enc_outputs = []
        for block in self.encoder:
            x = block(x)
            enc_outputs.append(x)

        x = self.middle_block(x)

        for i, (resblock, linear, layer_norm) in enumerate(self.decoder):
            x = torch.cat([enc_outputs.pop(), x], dim=-1)
            x = linear(x)

            if layer_norm is not None:
                x = layer_norm(x)

            x = resblock(x)

        return x
