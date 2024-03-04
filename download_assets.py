# MIT License

# Copyright (c) [2023] [Anima-Lab]


import os
from argparse import ArgumentParser
import requests
from tqdm import tqdm

_url_dict = {
    'imagenet512': 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz',
    'imagenet256': 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz',
    'vae': 'https://slurm-ord.s3.amazonaws.com/ckpts/autoencoder_kl.pth',
    'maskdit256-guidance': 'https://slurm-ord.s3.amazonaws.com/ckpts/256/imagenet256-ckpt-best_with_guidance.pt',
    'maskdit256-conditional': 'https://slurm-ord.s3.amazonaws.com/ckpts/256/imagenet256-ckpt-best_without_guidance.pt',
    'maskdit256-trained': 'https://slurm-ord.s3.amazonaws.com/ckpts/256/2000000.pt',
    'imagenet256-latent-lmdb': 'https://slurm-ord.s3.amazonaws.com/datasets/imagenet_256_latent_lmdb/train/',
    'inception': 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl',
    'maskdit512-guidance': 'https://slurm-ord.s3.amazonaws.com/ckpts/512/1080000.pt',
    'maskdit512-conditional': 'https://slurm-ord.s3.amazonaws.com/ckpts/512/1050000.pt',
    'imagenet512-latent-wds': 'https://slurm-ord.s3.amazonaws.com/datasets/imagenet-wds/',
}


def download_file(url, file_path):
    print('Start downloading...')
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=1024 * 1024 * 1024)):
                f.write(chunk)
    print('Complete')


def main(args):
    url = _url_dict[args.name]
    os.makedirs(args.dest, exist_ok=True)
    if args.name == 'imagenet512-latent-wds':
        num_files = 128
        for i in range(num_files):
            file_name = f'latent_imagenet_512_train-{i:04d}.tar'
            file_path = os.path.join(args.dest, file_name)
            download_file(url + file_name, file_path)
    elif args.name == 'imagenet256-latent-lmdb':
        file_lists = ['data.mdb', 'lock.mdb']
        for file_name in file_lists:
            file_path = os.path.join(args.dest, file_name)
            download_file(url + file_name, file_path)
    else:
        file_name = url.split('/')[-1]
        file_path = os.path.join(args.dest, file_name)
        download_file(url, file_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='Key of the asset')
    parser.add_argument('--dest', type=str, default='assets/fid_stats', help='Destination directory')
    args = parser.parse_args()
    main(args)