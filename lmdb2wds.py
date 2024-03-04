'''
This file implements the direction conversion from the latent ImageNet dataset to WebDataset. 
'''
import os
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import pickle

import webdataset as wds

from train_utils.datasets import ImageNetLatentDataset


def convert2wds(args):
    os.makedirs(args.outdir, exist_ok=True)
    wds_path = os.path.join(args.outdir, f'latent_imagenet_512_{args.split}-%04d.tar')
    dataset = ImageNetLatentDataset(args.datadir, resolution=args.resolution, num_channels=args.num_channels, split=args.split)

    with wds.ShardWriter(wds_path, maxcount=args.maxcount, maxsize=args.maxsize) as sink:
        for i in tqdm(range(len(dataset)), dynamic_ncols=True):
            if i % args.maxcount == 0:
                print(f'writing to the {i // args.maxcount}th shard')
            img, label = dataset[i]          # C, H, W
            label = np.argmax(label)         # int
            sink.write({'__key__': f'{i:07d}', 'latent': pickle.dumps(img), 'cls': label})


if __name__ == "__main__":
    parser = ArgumentParser('Convert the latent imagenet dataset to WebDataset')
    parser.add_argument('--maxcount', type=int, default=10010, help='max number of entries per shard')
    parser.add_argument('--maxsize', type=int, default=10 ** 10, help='max size per shard')
    parser.add_argument('--outdir', type=str, default='latent_imagenet_wds', help='path to save the converted dataset')
    parser.add_argument('--datadir', type=str, default='latent_imagenet', help='path to the latent imagenet dataset')
    parser.add_argument('--resolution', type=int, default=64, help='image resolution')
    parser.add_argument('--num_channels', type=int, default=8, help='number of image channels')
    parser.add_argument('--split', type=str, default='train', help='split of the dataset')
    args = parser.parse_args()
    convert2wds(args)