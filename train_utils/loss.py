# MIT License

# Copyright (c) [2023] [Anima-Lab]

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This code is adapted from https://github.com/NVlabs/edm/blob/main/training/loss.py. 
# The original code is licensed under a Creative Commons 
# Attribution-NonCommercial-ShareAlike 4.0 International License, which is can be found at LICENSE_EDM.txt. 

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
import torch.nn.functional as F

from utils import *


# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net,
                 images, 
                 labels=None, 
                 mask_ratio=0, 
                 mae_loss_coef=0, 
                 feat=None, augment_pipe=None):
        # sample x_t
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma

        model_out = net(y + n, sigma, labels, mask_ratio=mask_ratio, mask_dict=None, feat=feat)
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
        if mask_ratio == 0.0:
            loss += 0 * torch.sum(net.module.model.mask_token)
        assert loss.ndim == 1
        return loss


# ----------------------------------------------------------------------------


Losses = {
    'edm': EDMLoss
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
