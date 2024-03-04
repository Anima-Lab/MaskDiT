# MIT License

# Copyright (c) [2023] [Anima-Lab]


from argparse import ArgumentParser
import os
import json
from omegaconf import OmegaConf

import torch
from models.maskdit import Precond_models

from sample import generate_with_net
from utils import parse_float_none, parse_int_list, init_processes


def generate(args):
    rank = args.global_rank
    size = args.global_size
    config = OmegaConf.load(args.config)
    label_dict = json.load(open(args.label_dict, 'r'))
    class_label = label_dict[str(args.class_idx)][1]
    print(f'start sampling class {class_label}...')
    device = torch.device('cuda')
    # setup directory
    sample_dir = os.path.join(args.results_dir, class_label)
    os.makedirs(sample_dir, exist_ok=True)
    args.outdir = sample_dir
    # setup model
    model = Precond_models[config.model.precond](
        img_resolution=config.model.in_size,
        img_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        model_type=config.model.model_type,
        use_decoder=config.model.use_decoder,
        mae_loss_coef=config.model.mae_loss_coef,
        pad_cls_token=config.model.pad_cls_token,
        use_encoder_feat=config.model.self_cond,
    ).to(device)

    model.eval()
    print(f"{config.model.model_type} ((use_decoder: {config.model.use_decoder})) Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f'extras: {model.model.extras}, cls_token: {model.model.cls_token}')

    model = torch.compile(model)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt['ema'])
    generate_with_net(args, model, device, rank, size)

    print(f'sampling class {class_label} done!')


if __name__ == '__main__':
    parser = ArgumentParser('Sample from a trained model')
    # basic config
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument('--label_dict', type=str, default='assets/imagenet_label.json', help='path to label dict')
    parser.add_argument("--results_dir", type=str, default="samples", help='path to save samples')
    parser.add_argument('--ckpt_path', type=str, default=None, help='path to ckpt')

    # sampling
    parser.add_argument('--seeds', type=parse_int_list, default='100-131', help='Random seeds (e.g. 1,2,5-10)')
    parser.add_argument('--subdirs', action='store_true', help='Create subdirectory for every 1000 seeds')
    parser.add_argument('--class_idx', type=int, default=None, help='Class label  [default: random]')
    parser.add_argument("--cfg_scale", type=parse_float_none, default=None, help='None = no guidance, by default = 4.0')

    parser.add_argument('--num_steps', type=int, default=40, help='Number of sampling steps')
    parser.add_argument('--S_churn', type=int, default=0, help='Stochasticity strength')
    parser.add_argument('--solver', type=str, default=None, choices=['euler', 'heun'], help='Ablate ODE solver')
    parser.add_argument('--discretization', type=str, default=None, choices=['vp', 've', 'iddpm', 'edm'], help='Ablate ODE solver')
    parser.add_argument('--schedule', type=str, default=None, choices=['vp', 've', 'linear'], help='Ablate noise schedule sigma(t)')
    parser.add_argument('--scaling', type=str, default=None, choices=['vp', 'none'], help='Ablate signal scaling s(t)')
    parser.add_argument('--pretrained_path', type=str, default='assets/autoencoder_kl.pth', help='Autoencoder ckpt')

    parser.add_argument('--max_batch_size', type=int, default=32, help='Maximum batch size per GPU during sampling')
    parser.add_argument('--num_expected', type=int, default=32, help='Number of images to use')
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument('--fid_batch_size', type=int, default=32, help='Maximum batch size')

    # ddp 
    parser.add_argument('--num_proc_node', type=int, default=1, help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1, help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0, help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0, help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='localhost', help='address for master')
    args = parser.parse_args()
    args.global_rank = 0
    args.local_rank = 0
    args.global_size = 1
    init_processes(generate, args)