# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0,
                 dilemma_probability=0.0, token_drop_rate=0.0, dilemma_lambda=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T
        self.dilemma_probability = dilemma_probability
        self.token_drop_rate = token_drop_rate
        self.dilemma_lambda = dilemma_lambda

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        if dilemma_probability != 0.0:
            self.dilemma_head = nn.Linear(self.base_encoder.embed_dim, 1)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def get_pos(self, x, return_other_indices=False):
        N = x.size(0)
        L = self.base_encoder.patch_embed.num_patches
        len_keep = int(L * (1.0 - self.token_drop_rate))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        x_pos = ids_shuffle[:, :len_keep]
        if return_other_indices:
            return x_pos, ids_shuffle
        return x_pos

    def call_encoder(self, x):
        if self.dilemma_probability == 0.0:
            input_pos = None
            if self.token_drop_rate == 0.0:
                real_pos = None
            else:  # sparse, no dilemma
                real_pos = self.get_pos(x)
        else:
            real_pos, ids_shuffle = self.get_pos(x, return_other_indices=True)
            len_keep = real_pos.size(1)
            token_labels = torch.rand(real_pos.size(0), len_keep, device=x.device) < self.dilemma_probability
            if len_keep != x.size(1):
                noise = ids_shuffle[:, -len_keep:]
            else:
                noise = torch.flip(ids_shuffle, dims=[1])
            input_pos = torch.where(token_labels, noise, real_pos)
        cls, tokens = self.base_encoder(x, real_pos, input_pos)
        cls = self.predictor(cls)
        if self.dilemma_probability == 0.0:
            return cls, None
        return cls, F.binary_cross_entropy_with_logits(self.dilemma_head(tokens).squeeze(2), token_labels.float())

    def forward(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """
        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder
            # compute momentum features as targets
            k1, _ = self.momentum_encoder(x1)  # dense
            k2, _ = self.momentum_encoder(x2)

        # compute features
        q1, dilemma_loss1 = self.call_encoder(x1)
        q2, dilemma_loss2 = self.call_encoder(x2)
        con_loss = self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
        if self.dilemma_probability == 0.0:
            con_loss
        return con_loss + self.dilemma_lambda * (dilemma_loss1 + dilemma_loss2) / 2.0


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head  # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
