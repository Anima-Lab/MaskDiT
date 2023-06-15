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

# This code is adapted from https://github.com/NVlabs/edm/blob/main/generate.py. 
# The original code is licensed under a Creative Commons 
# Attribution-NonCommercial-ShareAlike 4.0 International License, which is can be found at LICENSE_EDM.txt. 

import argparse
import random

import PIL.Image
import lmdb
import numpy as np

import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from tqdm import tqdm

from models.maskdit import Precond_models, DiT_models
from utils import *
import autoencoder


# ----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

def edm_sampler(
        net, latents, class_labels=None, cfg_scale=None, feat=None, randn_like=torch.randn_like,
        num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat.float(), t_hat, class_labels, cfg_scale, feat=feat)['x'].to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next.float(), t_next, class_labels, cfg_scale, feat=feat)['x'].to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


# ----------------------------------------------------------------------------
# Generalized ablation sampler, representing the superset of all sampling
# methods discussed in the paper.

def ablation_sampler(
        net, latents, class_labels=None, cfg_scale=None, feat=None, randn_like=torch.randn_like,
        num_steps=18, sigma_min=None, sigma_max=None, rho=7,
        solver='heun', discretization='edm', schedule='linear', scaling='none',
        epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (
                sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.1, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.1, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device):  # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                    sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(
            t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat.float() / s(t_hat), sigma(t_hat), class_labels, cfg_scale, feat=feat)['x'].to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(
            t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime.float() / s(t_prime), sigma(t_prime), class_labels, cfg_scale, feat=feat)['x'].to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(
                t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next


# ----------------------------------------------------------------------------

def retrieve_n_features(batch_size, feat_path, feat_dim, num_classes, device, split='train', sample_mode='rand_full'):
    env = lmdb.open(os.path.join(feat_path, split), readonly=True, lock=False, create=False)

    # Start a new read transaction
    with env.begin() as txn:
        # Read all images in one single transaction, with one lock
        # We could split this up into multiple transactions if needed
        length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
        if sample_mode == 'rand_full':
            image_ids = random.sample(range(length // 2), batch_size)
            image_ids_y = image_ids
        elif sample_mode == 'rand_repeat':
            image_ids = random.sample(range(length // 2), 1) * batch_size
            image_ids_y = image_ids
        elif sample_mode == 'rand_y':
            image_ids = random.sample(range(length // 2), 1) * batch_size
            image_ids_y = random.sample(range(length // 2), batch_size)
        else:
            raise NotImplementedError
        features, labels = [], []
        for image_id, image_id_y in zip(image_ids, image_ids_y):
            feat_bytes = txn.get(f'feat-{str(image_id)}'.encode('utf-8'))
            y_bytes = txn.get(f'y-{str(image_id_y)}'.encode('utf-8'))
            feat = np.frombuffer(feat_bytes, dtype=np.float32).reshape([feat_dim]).copy()
            y = int(y_bytes.decode('utf-8'))
            features.append(feat)
            labels.append(y)
        features = torch.from_numpy(np.stack(features)).to(device)
        labels = torch.from_numpy(np.array(labels)).to(device)
        class_labels = torch.zeros([batch_size, num_classes], device=device)
        if num_classes > 0:
            class_labels = torch.eye(num_classes, device=device)[labels]
        assert features.shape[0] == class_labels.shape[0] == batch_size
    return features, class_labels



@torch.no_grad()
def generate_with_net(args, net, device, feat_path=None, ext_feature_dim=0):
    rank = args.global_rank
    size = args.global_size
    seeds = args.seeds
    num_batches = ((len(seeds) - 1) // (args.max_batch_size * size) + 1) * size
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[rank:: size]

    net.eval()

    # Setup sampler
    sampler_kwargs = dict(num_steps=args.num_steps, S_churn=args.S_churn,
                          solver=args.solver, discretization=args.discretization,
                          schedule=args.schedule, scaling=args.scaling)
    sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
    have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
    sampler_fn = ablation_sampler if have_ablation_kwargs else edm_sampler
    mprint(f"sampler_kwargs: {sampler_kwargs}, \nsampler fn: {sampler_fn.__name__}")
    # Setup autoencoder
    vae = autoencoder.get_model(args.pretrained_path).to(device)

    # generate images
    mprint(f'Generating {len(seeds)} images to "{args.outdir}"...')
    for batch_seeds in tqdm(rank_batches, unit='batch', disable=(rank != 0)):
        dist.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = torch.zeros([batch_size, net.num_classes], device=device)
        if net.num_classes:
            class_labels = torch.eye(net.num_classes, device=device)[
                rnd.randint(net.num_classes, size=[batch_size], device=device)]
        if args.class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, args.class_idx] = 1

        # retrieve features from training set [support random only]
        feat = None
        if feat_path is not None and os.path.isdir(feat_path):
            feat, class_labels = retrieve_n_features(batch_size, feat_path, ext_feature_dim,
                                                     net.num_classes, device, sample_mode=args.sample_mode)

        # Generate images.
        def recur_decode(z):
            try:
                return vae.decode(z)
            except:  # reduce the batch for vae decoder but two forward passes when OOM happens occasionally
                assert z.shape[2] % 2 == 0
                z1, z2 = z.tensor_split(2)
                return torch.cat([recur_decode(z1), recur_decode(z2)])

        with torch.no_grad():
            z = sampler_fn(net, latents.float(), class_labels.float(), randn_like=rnd.randn_like,
                           cfg_scale=args.cfg_scale, feat=feat, **sampler_kwargs).float()
            images = recur_decode(z)
            
        # Save images.
        images_np = images.add_(1).mul(127.5).clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        # images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(args.outdir, f'{seed - seed % 1000:06d}') if args.subdirs else args.outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)


def generate(args):
    device = torch.device("cuda")

    mprint(f'cf_scale: {args.cfg_scale}')
    if args.global_rank == 0:
        os.makedirs(args.outdir, exist_ok=True)
        logger = Logger(file_name=f'{args.outdir}/log.txt', file_mode="a+", should_flush=True)

    # Create model:
    net = Precond_models[args.precond](
        img_resolution=args.image_size,
        img_channels=args.image_channels,
        num_classes=args.num_classes,
        model_type=args.model_type,
        use_decoder=args.use_decoder,
        mae_loss_coef=args.mae_loss_coef,
        pad_cls_token=args.pad_cls_token,
        ext_feature_dim=args.ext_feature_dim
    ).to(device)
    mprint(
        f"{args.model_type} (use_decoder: {args.use_decoder}) Model Parameters: {sum(p.numel() for p in net.parameters()):,}")

    # Load checkpoints
    ckpt = torch.load(args.ckpt_path, map_location=device)
    net.load_state_dict(ckpt['ema'])
    mprint(f'Load weights from {args.ckpt_path}')

    generate_with_net(args, net, device)

    # Done.
    cleanup()
    if args.global_rank == 0:
        logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('sampling parameters')

    # ddp
    parser.add_argument('--num_proc_node', type=int, default=1, help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1, help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0, help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0, help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='localhost', help='address for master')

    # sampling
    parser.add_argument("--feat_path", type=str, default='')
    parser.add_argument("--ext_feature_dim", type=int, default=0)
    parser.add_argument('--ckpt_path', type=str, required=True, help='Network pickle filename')
    parser.add_argument('--outdir', type=str, required=True, help='sampling results save filename')
    parser.add_argument('--seeds', type=parse_int_list, default='0-63', help='Random seeds (e.g. 1,2,5-10)')
    parser.add_argument('--subdirs', action='store_true', help='Create subdirectory for every 1000 seeds')
    parser.add_argument('--class_idx', type=int, default=None, help='Class label  [default: random]')
    parser.add_argument('--max_batch_size', type=int, default=64, help='Maximum batch size per GPU')

    parser.add_argument("--cfg_scale", type=parse_float_none, default=None, help='None = no guidance, by default = 4.0')

    parser.add_argument('--num_steps', type=int, default=18, help='Number of sampling steps')
    parser.add_argument('--S_churn', type=int, default=0, help='Stochasticity strength')
    parser.add_argument('--solver', type=str, default=None, choices=['euler', 'heun'], help='Ablate ODE solver')
    parser.add_argument('--discretization', type=str, default=None, choices=['vp', 've', 'iddpm', 'edm'],
                        help='Ablate ODE solver')
    parser.add_argument('--schedule', type=str, default=None, choices=['vp', 've', 'linear'],
                        help='Ablate noise schedule sigma(t)')
    parser.add_argument('--scaling', type=str, default=None, choices=['vp', 'none'], help='Ablate signal scaling s(t)')
    parser.add_argument('--pretrained_path', type=str, default='assets/stable_diffusion/autoencoder_kl.pth',
                        help='Autoencoder ckpt')

    # model
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--image_channels", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=1000, help='0 means unconditional')
    parser.add_argument("--model_type", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument('--precond', type=str, choices=['vp', 've', 'edm'], default='edm', help='precond train & loss')
    parser.add_argument("--use_decoder", type=str2bool, default=False)
    parser.add_argument("--pad_cls_token", type=str2bool, default=False)
    parser.add_argument('--mae_loss_coef', type=float, default=0, help='0 means no MAE loss')
    parser.add_argument('--sample_mode', type=str, default='rand_full', help='[rand_full, rand_repeat]')

    args = parser.parse_args()
    args.global_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            args.global_rank = rank + args.node_rank * args.num_process_per_node
            p = Process(target=init_processes, args=(generate, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        print('Single GPU run')
        assert args.global_size == 1 and args.local_rank == 0
        args.global_rank = 0
        init_processes(generate, args)
