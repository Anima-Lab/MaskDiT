# MIT License

# Copyright (c) [2023] [Anima-Lab]


import os
from argparse import ArgumentParser
import requests
from tqdm import tqdm

_url_dict = {
    'imagenet256': 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz', 
    'imagenet128': 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/128/VIRTUAL_imagenet128_labeled.npz',
    'imagenet64': 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/64/VIRTUAL_imagenet64_labeled.npz',
    'vae': 'https://maskdit-bucket.s3.us-west-2.amazonaws.com/autoencoder_kl.pth',
    'maskdit-unmask-finetune': 'https://maskdit-bucket.s3.us-west-2.amazonaws.com/imagenet256-ckpt-best_with_guidance.pt',
    'maskdit-cos-finetune': 'https://maskdit-bucket.s3.us-west-2.amazonaws.com/imagenet256-ckpt-best_without_guidance.pt',
    'maskdit-trained': 'https://maskdit-bucket.s3.us-west-2.amazonaws.com/2000000.pt', 
    'imagenet-latent-data': 'https://maskdit-bucket.s3.us-west-2.amazonaws.com/imagenet_256_latent_lmdb.zip',
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
    file_name = url.split('/')[-1]
    os.makedirs(args.dest, exist_ok=True)
    file_path = os.path.join(args.dest, file_name)
    download_file(url, file_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, required=True, 
                        choices=['imagenet256', 'imagenet128', 'imagenet64', 'vae', 'maskdit-unmask-finetune', 'maskdit-cos-finetune', 'maskdit-trained', 'imagenet-latent-data'])
    parser.add_argument('--dest', type=str, default='assets/fid_stats', help='Destination directory')
    args = parser.parse_args()
    main(args)