# MIT License

# Copyright (c) [2023] [Anima-Lab]

from argparse import ArgumentParser
import os
from collections import OrderedDict
from omegaconf import OmegaConf

import torch

import accelerate

from fid import calc
from models.maskdit import Precond_models
from sample import generate_with_net
from utils import dist, mprint, get_ckpt_paths, Logger, parse_int_list, parse_float_none


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


def eval_fn(model, args, device, rank, size):
    generate_with_net(args, model, device, rank, size)
    dist.barrier()
    fid = calc(args.outdir, args.ref_path, args.num_expected, args.global_seed, args.fid_batch_size)
    mprint(f'{args.num_expected} samples generated and saved in {args.outdir}')
    mprint(f'guidance: {args.cfg_scale} FID: {fid}')
    dist.barrier()
    return fid


def eval_loop(args):
    config = OmegaConf.load(args.config)
    accelerator = accelerate.Accelerator()

    device = accelerator.device
    size = accelerator.num_processes
    rank = accelerator.process_index
    print(f'world_size: {size}, rank: {rank}')
    experiment_dir = args.exp_dir
    
    if accelerator.is_main_process:
        logger = Logger(file_name=f'{experiment_dir}/log_eval.txt', file_mode="a+", should_flush=True)
        # setup wandb

    model = Precond_models[config.model.precond](
        img_resolution=config.model.in_size,
        img_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        model_type=config.model.model_type,
        use_decoder=config.model.use_decoder,
        mae_loss_coef=config.model.mae_loss_coef,
        pad_cls_token=config.model.pad_cls_token,
    ).to(device)
    # Note that parameter initialization is done within the model constructor
    model.eval()
    mprint(f"{config.model.model_type} ((use_decoder: {config.model.use_decoder})) Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    mprint(f'extras: {model.model.extras}, cls_token: {model.model.cls_token}')

    # model = torch.compile(model)
    # Load checkpoints
    mprint('start evaluating...')

    args.outdir = os.path.join(experiment_dir, 'fid', f'edm-steps{args.num_steps}_cfg{args.cfg_scale}')
    os.makedirs(args.outdir, exist_ok=True)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['ema'])
    fid = eval_fn(model, args, device, rank, size)
    mprint(f'FID: {fid}')

    if accelerator.is_main_process:
        logger.close()
    accelerator.end_training()


if __name__ == '__main__':
    parser = ArgumentParser('training parameters')
    # basic config
    parser.add_argument('--config', type=str, required=True, help='path to config file')

    # training
    parser.add_argument("--exp_dir", type=str, required=True, help='The exp directory to evaluate, it must contain a checkpoints folder')
    parser.add_argument('--ckpt', type=str, required=True, help='path to the checkpoint')

    # sampling
    parser.add_argument('--seeds', type=parse_int_list, default='100000-149999', help='Random seeds (e.g. 1,2,5-10)')
    parser.add_argument('--subdirs', action='store_true', help='Create subdirectory for every 1000 seeds')
    parser.add_argument('--class_idx', type=int, default=None, help='Class label  [default: random]')
    parser.add_argument('--max_batch_size', type=int, default=50, help='Maximum batch size per GPU during sampling, must be a factor of 50k if torch.compile is used')
    parser.add_argument("--cfg_scale", type=parse_float_none, default=None, help='None = no guidance, by default = 4.0')

    parser.add_argument('--num_steps', type=int, default=40, help='Number of sampling steps')
    parser.add_argument('--S_churn', type=int, default=0, help='Stochasticity strength')
    parser.add_argument('--solver', type=str, default=None, choices=['euler', 'heun'], help='Ablate ODE solver')
    parser.add_argument('--discretization', type=str, default=None, choices=['vp', 've', 'iddpm', 'edm'], help='Ablate ODE solver')
    parser.add_argument('--schedule', type=str, default=None, choices=['vp', 've', 'linear'], help='Ablate noise schedule sigma(t)')
    parser.add_argument('--scaling', type=str, default=None, choices=['vp', 'none'], help='Ablate signal scaling s(t)')
    parser.add_argument('--pretrained_path', type=str, default='assets/stable_diffusion/autoencoder_kl.pth', help='Autoencoder ckpt')

    parser.add_argument('--ref_path', type=str, default='assets/fid_stats/VIRTUAL_imagenet512.npz', help='Dataset reference statistics')
    parser.add_argument('--num_expected', type=int, default=50000, help='Number of images to use')
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument('--fid_batch_size', type=int, default=128, help='Maximum batch size per GPU')

    args = parser.parse_args()
    
    torch.backends.cudnn.benchmark = True
    eval_loop(args)
