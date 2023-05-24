# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch
import torch.nn.functional as F

from utils import *


# ----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5, class_dropout_prob=0.0):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t
        self.class_dropout_prob = class_dropout_prob

    def __call__(self, net, images, labels, mask_ratio=0, mae_loss_coef=0, feat=None, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        if self.class_dropout_prob > 0 and np.random.rand() < self.class_dropout_prob:
            labels = labels * 0.0
            mask_ratio = 0
        model_out = net(y + n, sigma, labels, mask_ratio=mask_ratio, feat=feat, augment_labels=augment_labels)
        D_yn = model_out['x']
        loss = weight * ((D_yn - y) ** 2)  # (N, C, H, W)
        if mask_ratio > 0:
            assert net.training and 'mask' in model_out
            loss = F.avg_pool2d(loss.mean(dim=1), net.module.model.patch_size).flatten(1)  # (N, L)
            unmask = 1 - model_out['mask']
            loss = (loss * unmask).sum(dim=1) / unmask.sum(dim=1)  # (N)
            if mae_loss_coef > 0:
                loss += mae_loss_coef * mae_loss(net.module, y + n, D_yn, 1 - unmask)
        else:
            loss = mean_flat(loss)  # (N)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()


# ----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100, class_dropout_prob=0.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.class_dropout_prob = class_dropout_prob

    def __call__(self, net, images, labels, mask_ratio=0, mae_loss_coef=0, feat=None, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        if self.class_dropout_prob > 0 and np.random.rand() < self.class_dropout_prob:
            labels = labels * 0.0
            mask_ratio = 0
        model_out = net(y + n, sigma, labels, mask_ratio=mask_ratio, feat=feat, augment_labels=augment_labels)
        D_yn = model_out['x']
        loss = weight * ((D_yn - y) ** 2)  # (N, C, H, W)
        if mask_ratio > 0:
            assert net.training and 'mask' in model_out
            loss = F.avg_pool2d(loss.mean(dim=1), net.module.model.patch_size).flatten(1)  # (N, L)
            unmask = 1 - model_out['mask']
            loss = (loss * unmask).sum(dim=1) / unmask.sum(dim=1)  # (N)
            if mae_loss_coef > 0:
                loss += mae_loss_coef * mae_loss(net.module, y + n, D_yn, 1 - unmask)
        else:
            loss = mean_flat(loss)  # (N)
        return loss


# ----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, class_dropout_prob=0.0, uncond_mask_ratio=0.0):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.class_dropout_prob = class_dropout_prob
        self.uncond_mask_ratio = uncond_mask_ratio

    def __call__(self, net, images, labels=None, mask_ratio=0, mae_loss_coef=0, feat=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        # set lable to None with probability self.class_dropout_prob
        if self.class_dropout_prob > 0 and np.random.rand() < self.class_dropout_prob:
            labels = labels * 0.0
            mask_ratio = self.uncond_mask_ratio
        model_out = net(y + n, sigma, labels, mask_ratio=mask_ratio, feat=feat, augment_labels=augment_labels)
        D_yn = model_out['x']
        assert D_yn.shape == y.shape
        loss = weight * ((D_yn - y) ** 2)  # (N, C, H, W)
        if mask_ratio > 0:
            assert net.training and 'mask' in model_out
            loss = F.avg_pool2d(loss.mean(dim=1), net.module.model.patch_size).flatten(1)  # (N, L)
            unmask = 1 - model_out['mask']
            loss = (loss * unmask).sum(dim=1) / unmask.sum(dim=1)  # (N)
            assert loss.ndim == 1
            if mae_loss_coef > 0:
                loss += mae_loss_coef * mae_loss(net.module, y + n, D_yn, 1 - unmask)
        else:
            loss = mean_flat(loss)  # (N)
        assert loss.ndim == 1
        if self.uncond_mask_ratio == 0.0:
            loss += 0 * torch.sum(net.module.model.mask_token)
        return loss

# ----------------------------------------------------------------------------


Losses = {
    'vp': VPLoss, 've': VELoss, 'edm': EDMLoss
}


# ----------------------------------------------------------------------------

def patchify(imgs, patch_size=2, num_channels=4):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p, c = patch_size, num_channels
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
    return x


def mae_loss(net, target, pred, mask, norm_pix_loss=True):
    target = patchify(target, net.model.patch_size, net.model.out_channels)
    pred = patchify(pred, net.model.patch_size, net.model.out_channels)
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum(dim=1) / mask.sum(dim=1)  # mean loss on removed patches, (N)
    assert loss.ndim == 1
    return loss

# def mask_ratio_schedule(sigma, ln_sigma_min=-4.8, ln_sigma_max=2.4, mask_ratio_min=0, mask_ratio_max=0.8):
#     # log-linear schedule
#
#     A = (mask_ratio_max - mask_ratio_min) / (ln_sigma_min - ln_sigma_max)
#     B = (mask_ratio_min * ln_sigma_min - mask_ratio_max * ln_sigma_max) / (ln_sigma_min - ln_sigma_max)
#     mask_ratio = np.clip(A * np.log(sigma) + B, mask_ratio_min, mask_ratio_max)
#     return mask_ratio


