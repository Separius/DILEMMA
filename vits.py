# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg, checkpoint_seq

__all__ = [
    'vit_small',
    'vit_base',
]


class VisionTransformerMoCo(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.num_prefix_tokens == 1 and self.no_embed_class and self.global_pool == 'token'

    @staticmethod
    def select_subset(x, pos):
        if pos is None:
            return x
        return torch.gather(x, dim=1, index=pos.unsqueeze(-1).expand(x.size(0), -1, x.size(-1)))

    def _pos_embed(self, x, real_pos=None, input_pos=None):
        x = self.select_subset(x, real_pos)
        x = x + self.select_subset(self.pos_embed, real_pos if input_pos is None else input_pos)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        return self.pos_drop(x)

    def forward_features(self, x, real_pos=None, input_pos=None):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x, real_pos=None, input_pos=None):
        x = self.forward_features(x, real_pos, input_pos)
        tokens = x[:, self.num_prefix_tokens:]
        x = self.forward_head(x)
        return x, tokens


def vit_small(**kwargs):
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, no_embed_class=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool='token', **kwargs)
    model.default_cfg = _cfg()
    return model


def vit_base(**kwargs):
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, no_embed_class=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool='token', **kwargs)
    model.default_cfg = _cfg()
    return model
