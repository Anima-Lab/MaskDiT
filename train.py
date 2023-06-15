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

import argparse
import os.path
from collections import OrderedDict
from copy import deepcopy
from time import time
from omegaconf import OmegaConf


import apex
import torch
import wandb
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
from torchvision import transforms

from fid import calc
from models.maskdit import Precond_models
from train_utils.loss import Losses
from train_utils.datasets import imagenet_lmdb_dataset
from train_utils.helper import get_mask_ratio_fn

from sample import generate_with_net
from utils import dist, mprint, get_latest_ckpt, Logger, \
    ddp_sync, init_processes, cleanup, \
    str2bool, parse_str_none, parse_int_list, parse_float_none

from autoencoder import get_model

# ------------------------------------------------------------
# Training Helper Function

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

# ------------------------------------------------------------


def train_loop(args):
    # load configuration
    config = OmegaConf.load(args.config)
    if config.train.tf32:
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.allow_tf32 = True
    mprint('start training...')
    size = args.global_size
    rank = args.global_rank
    print(f'global_rank: {rank}, global_size: {size}, local_rank: {args.local_rank}')
    print(f'torch.cuda.device_count(): {torch.cuda.device_count()}')
    print(f'dist.get_world_size(): {dist.get_world_size()}, dist.get_rank(): {dist.get_rank()}')
    device = torch.device("cuda")

    seed = args.global_rank * args.num_workers + args.global_seed 
    torch.manual_seed(seed)
    enable_amp = not args.no_amp
    mprint(f"enable_amp: {enable_amp}, TF32: {config.train.tf32}")
    # Select batch size per GPU
    num_accumulation_rounds = config.train.grad_accum
    micro_batch = config.train.batchsize
    batch_gpu_total = micro_batch * num_accumulation_rounds
    global_batch_size = batch_gpu_total * size
    mprint(f"Global batchsize: {global_batch_size},  batchsize per GPU: {batch_gpu_total}, micro_batch: {micro_batch}.")

    class_dropout_prob = config.model.class_dropout_prob
    log_every = config.log.log_every
    ckpt_every = config.log.ckpt_every
    mask_ratio_fn = get_mask_ratio_fn(config.model.mask_ratio_fn, 
                                      config.model.mask_ratio, config.model.mask_ratio_min)

    # Setup an experiment folder
    model_name = config.model.model_type.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    data_name = config.data.dataset
    if args.ckpt_path is not None and args.use_ckpt_path:  # use the existing exp path (mainly used for fine-tuning)
        checkpoint_dir = os.path.dirname(args.ckpt_path)
        experiment_dir = os.path.dirname(checkpoint_dir)
        exp_name = os.path.basename(experiment_dir)
    else:  # start a new exp path (and resume from the latest checkpoint if possible)
        cond_gen = 'cond' if config.model.num_classes else 'uncond'
        exp_name = f'{model_name}-{config.model.precond}-{data_name}-{cond_gen}-m{config.model.mask_ratio}-de{int(config.model.use_decoder)}' \
                   f'-mae{config.model.mae_loss_coef}-bs-{global_batch_size}-lr{config.train.lr}{args.tag}'
        experiment_dir = f"{args.results_dir}/{exp_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        if args.ckpt_path is None:
            args.ckpt_path = get_latest_ckpt(checkpoint_dir)  # Resumes from the latest checkpoint if it exists
    mprint(f"Experiment directory created at {experiment_dir}")

    if rank == 0:
        logger = Logger(file_name=f'{experiment_dir}/log.txt', file_mode="a+", should_flush=True)
        # setup wandb
        if args.use_wandb:
            wandb.init(entity=config.wandb.entity,
                       project=config.wandb.project,
                       group=config.wandb.group,
                       config=OmegaConf.to_container(config),
                       reinit=True,
                       settings=wandb.Settings(start_method='fork'))

    # Setup dataset
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = imagenet_lmdb_dataset(config.data.root, transform=transform, resolution=config.data.resolution)
    sampler = DistributedSampler(
        dataset, num_replicas=size, rank=rank, shuffle=True, seed=args.global_seed
    )
    loader = DataLoader(
        dataset, batch_size=batch_gpu_total, shuffle=False,
        sampler=sampler, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=True,
        drop_last=True
    )
    mprint(f"Dataset contains {len(dataset):,} images ({config.data.root})")

    steps_per_epoch = len(dataset) // global_batch_size
    mprint(f"{steps_per_epoch} steps per epoch")

    # Create model:

    vae = get_model(args.pretrained_path).to(device)
    vae = torch.compile(vae)
    assert config.model.in_size * 8 == config.data.resolution
    model = Precond_models[config.model.precond](
        img_resolution=config.model.in_size,
        img_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        model_type=config.model.model_type,
        use_decoder=config.model.use_decoder,
        mae_loss_coef=config.model.mae_loss_coef,
        pad_cls_token=config.model.pad_cls_token,
    )
    # Note that parameter initialization is done within the model constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    mprint(f"{config.model.model_type} ((use_decoder: {config.model.use_decoder})) Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    mprint(f'extras: {model.model.extras}, cls_token: {model.model.cls_token}')

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=config.train.lr, adam_w_mode=True, weight_decay=0)
    model = torch.compile(model)
    ema = torch.compile(ema)
    # Load checkpoints
    train_steps_start = 0
    epoch_start = 0

    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'], strict=args.use_strict_load)
        ema.load_state_dict(ckpt['ema'], strict=args.use_strict_load)
        mprint(f'Load weights from {args.ckpt_path}')
        if args.use_strict_load:
            optimizer.load_state_dict(ckpt['opt'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        mprint(f'Load optimizer state..')
        train_steps_start = int(os.path.basename(args.ckpt_path).split('.pt')[0])
        epoch_start = train_steps_start // steps_per_epoch
        mprint(f"train_steps_start: {train_steps_start}")
        del ckpt # conserve memory

        # FID evaluation for the loaded weights
        if args.enable_eval:
            start_time = time()
            args.outdir = os.path.join(experiment_dir, 'fid', f'edm-steps{args.num_steps}-ckpt{train_steps_start}_cfg{args.cfg_scale}')
            os.makedirs(args.outdir, exist_ok=True)
            generate_with_net(args, ema, device)
            dist.barrier()
            fid = calc(args.outdir, config.eval.ref_path, args.num_expected, args.global_seed, args.fid_batch_size)
            mprint(f"time for fid calc: {time() - start_time}")
            if rank == 0 and args.use_wandb:
                wandb.log({f'fid': fid}, step=train_steps_start)
            dist.barrier()

    model = DDP(model.to(device), device_ids=[device])

    # Setup loss
    loss_fn = Losses[config.model.precond]()

    # Prepare models for training:
    if args.ckpt_path is None:
        assert train_steps_start == 0
        update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)

    # Variables for monitoring/logging purposes:
    train_steps = train_steps_start
    log_steps = 0
    running_loss = 0
    start_time = time()
    label_table = torch.eye(config.model.num_classes, device=device)
    mprint(f"Training for {config.train.epochs} epochs...")
    for epoch in range(epoch_start, config.train.epochs):
        sampler.set_epoch(epoch)
        mprint(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            with torch.no_grad():
                # map input image to latent space and normalize it. 
                x = vae.encode(x)
            # x = torch.randn([x.shape[0], 4, 32, 32], device=device)
            y = label_table[y.to(device)]

            # Accumulate gradients.
            loss_batch = 0
            model.zero_grad(set_to_none=True)
            curr_mask_ratio = mask_ratio_fn((train_steps - train_steps_start) / config.train.max_num_steps)
            for round_idx in range(num_accumulation_rounds):
                with ddp_sync(model, (round_idx == num_accumulation_rounds - 1)):
                    x_ = x[round_idx * micro_batch: (round_idx + 1) * micro_batch]
                    y_ = y[round_idx * micro_batch: (round_idx + 1) * micro_batch]
                    if class_dropout_prob > 0:  # unconditional training with probability class_dropout_prob
                        y_ = y_ * (torch.rand([y_.shape[0], 1], device=device) >= class_dropout_prob).to(y.dtype)

                    with torch.autocast(device_type="cuda", enabled=enable_amp):
                        loss = loss_fn(net=model, images=x_, labels=y_, 
                                       mask_ratio=curr_mask_ratio,
                                       mae_loss_coef=config.model.mae_loss_coef)
                        loss_mean = loss.sum().mul(1 / batch_gpu_total)
                    # loss_mean.backward()
                    scaler.scale(loss_mean).backward()
                    loss_batch += loss_mean.item()

            # Update weights with lr warmup.
            lr_cur = config.train.lr * min(train_steps * global_batch_size / max(config.train.lr_rampup_kimg * 1000, 1e-8), 1)
            for g in optimizer.param_groups:
                g['lr'] = lr_cur
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss_batch
            log_steps += 1
            train_steps += 1
            if train_steps > (train_steps_start + config.train.max_num_steps):
                break
            if train_steps % log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / size
                mprint(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if rank == 0 and args.use_wandb:
                    wandb.log({f'train Loss': avg_loss, 'lr': lr_cur}, step=train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint:
            if train_steps % ckpt_every == 0 and train_steps > train_steps_start:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    mprint(f"Saved checkpoint to {checkpoint_path}")
                    del checkpoint  # conserve memory
                dist.barrier()

                # FID evaluation during training
                if args.enable_eval:
                    start_time = time()
                    args.outdir = os.path.join(experiment_dir, 'fid', f'edm-steps{args.num_steps}-ckpt{train_steps}_cfg{args.cfg_scale}')
                    os.makedirs(args.outdir, exist_ok=True)
                    generate_with_net(args, ema, vae, device)
                    dist.barrier()
                    fid = calc(args.outdir, args.ref_path, args.num_expected, args.global_seed, args.fid_batch_size)
                    mprint(f"time for fid calc: {time() - start_time}, fid: {fid}")
                    if rank == 0 and args.use_wandb:
                        wandb.log({f'fid': fid}, step=train_steps)
                    dist.barrier()
                start_time = time()
                
    cleanup()
    if rank == 0:
        logger.close()
        if args.use_wandb:
            wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('training parameters')
    # basic config
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    # ddp
    parser.add_argument('--num_proc_node', type=int, default=1, help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1, help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0, help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0, help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='localhost', help='address for master')

    # training
    parser.add_argument("--feat_path", type=str, default='')
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--ckpt_path", type=parse_str_none, default=None)

    parser.add_argument('--xflip', action='store_true', help='enable xflip for data')

    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--no_amp', action='store_true', help="Disable automatic mixed precision.")

    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50_000)
    parser.add_argument("--use_wandb", action='store_true', help='enable wandb logging')
    parser.add_argument("--use_ckpt_path", type=str2bool, default=True)
    parser.add_argument("--use_strict_load", type=str2bool, default=True)
    parser.add_argument("--tag", type=str, default='')

    # sampling
    parser.add_argument('--enable_eval', action='store_true', help='enable fid calc during training')
    parser.add_argument('--seeds', type=parse_int_list, default='0-49999', help='Random seeds (e.g. 1,2,5-10)')
    parser.add_argument('--subdirs', action='store_true', help='Create subdirectory for every 1000 seeds')
    parser.add_argument('--class_idx', type=int, default=None, help='Class label  [default: random]')
    parser.add_argument('--max_batch_size', type=int, default=128, help='Maximum batch size per GPU during sampling')

    parser.add_argument("--cfg_scale", type=parse_float_none, default=None, help='None = no guidance, by default = 4.0')

    parser.add_argument('--num_steps', type=int, default=40, help='Number of sampling steps')
    parser.add_argument('--S_churn', type=int, default=0, help='Stochasticity strength')
    parser.add_argument('--solver', type=str, default=None, choices=['euler', 'heun'], help='Ablate ODE solver')
    parser.add_argument('--discretization', type=str, default=None, choices=['vp', 've', 'iddpm', 'edm'], help='Ablate ODE solver')
    parser.add_argument('--schedule', type=str, default=None, choices=['vp', 've', 'linear'], help='Ablate noise schedule sigma(t)')
    parser.add_argument('--scaling', type=str, default=None, choices=['vp', 'none'], help='Ablate signal scaling s(t)')
    parser.add_argument('--pretrained_path', type=str, default='assets/stable_diffusion/autoencoder_kl.pth', help='Autoencoder ckpt')

    parser.add_argument('--ref_path', type=str, default='assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz', help='Dataset reference statistics')
    parser.add_argument('--num_expected', type=int, default=50000, help='Number of images to use')
    parser.add_argument('--fid_batch_size', type=int, default=256, help='Maximum batch size per GPU per GPU')

    args = parser.parse_args()
    args.global_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node
    
    torch.backends.cudnn.benchmark = True
    mp.set_start_method('spawn', force=True)
    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            args.global_rank = rank + args.node_rank * args.num_process_per_node
            # print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, args.local_rank, args.global_rank))
            p = mp.Process(target=init_processes, args=(train_loop, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        print('Single GPU run')
        assert args.global_size == 1 and args.local_rank == 0
        args.global_rank = 0
        init_processes(train_loop, args)
