import argparse
import os, sys
import time

import io
import lmdb

import numpy as np
from PIL import Image

import torch
import torchvision
from torch import nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor


################################################################################
# ImageNet - LMDB
###############################################################################

def lmdb_loader(path, lmdb_data, resolution):
    # In-memory binary streams
    with lmdb_data.begin(write=False, buffers=True) as txn:
        bytedata = txn.get(path.encode('ascii'))
    img = Image.open(io.BytesIO(bytedata)).convert('RGB')
    arr = center_crop_arr(img, resolution)
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])  # CHW
    return arr


def imagenet_lmdb_dataset(
        root, transform=None, target_transform=None, resolution=256,
        loader=lmdb_loader):
    """
    You can create this dataloader using:
    train_data = imagenet_lmdb_dataset(traindir, transform=train_transform)
    valid_data = imagenet_lmdb_dataset(validdir, transform=val_transform)
    """

    if root.endswith('/'):
        root = root[:-1]
    pt_path = os.path.join(
        root + '_faster_imagefolder.lmdb.pt')
    lmdb_path = os.path.join(
        root + '_faster_imagefolder.lmdb')
    if os.path.isfile(pt_path) and os.path.isdir(lmdb_path):
        print('Loading pt {} and lmdb {}'.format(pt_path, lmdb_path))
        data_set = torch.load(pt_path)
    else:
        data_set = ImageFolder(
            root, None, None, None)
        torch.save(data_set, pt_path, pickle_protocol=4)
        print('Saving pt to {}'.format(pt_path))
        print('Building lmdb to {}'.format(lmdb_path))
        env = lmdb.open(lmdb_path, map_size=1e12)
        with env.begin(write=True) as txn:
            for path, class_index in data_set.imgs:
                with open(path, 'rb') as f:
                    data = f.read()
                txn.put(path.encode('ascii'), data)
    data_set.lmdb_data = lmdb.open(
        lmdb_path, readonly=True, max_readers=1, lock=False, readahead=False,
        meminit=False)
    # reset transform and target_transform
    data_set.samples = data_set.imgs
    data_set.transform = transform
    data_set.target_transform = target_transform
    data_set.loader = lambda path: loader(path, data_set.lmdb_data, resolution)

    return data_set


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='imagenet', type=str)
    parser.add_argument('--resolution', default=256, type=int)
    parser.add_argument('--arch', default='resnet50', type=str)
    parser.add_argument('--pretrained', default='assets/moco_v3/r-50-1000ep.pth.tar', type=str, help='path to moco pretrained checkpoint')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--split', default='train', type=str)
    parser.add_argument('--outdir', type=str, default='data/imagenet_256_latent_lmdb', help='output directory')
    args = parser.parse_args()

    assert args.split in ['train', 'val']
    dataset = imagenet_lmdb_dataset(root=f'/datasets/imagenet_lmdb/{args.split}')
    print(f'data size: {len(dataset)}')

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = torchvision.models.__dict__[args.arch]()
    linear_keyword = 'fc'

    if os.path.isfile(args.pretrained):
        print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                # remove prefix
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}

        print("=> loaded pre-trained model '{}'".format(args.pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(args.pretrained))

    model_feature = create_feature_extractor(model, return_nodes=['avgpool'])

    model_feature = nn.DataParallel(model_feature)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_feature.to(device)

    def extract_feature():
        outdir = f'data/{args.data_name}_{args.resolution}_moco_v3_rn50_lmdb'
        target_db_dir = os.path.join(outdir, args.split)
        os.makedirs(target_db_dir, exist_ok=True)
        target_env = lmdb.open(target_db_dir, map_size=pow(2,40), readahead=False)

        dataset_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                    num_workers=8, pin_memory=True, persistent_workers=True)

        idx = 0
        begin = time.time()
        for batch in dataset_loader:
            img, label = batch
            assert img.min() >= -1 and img.max() <= 1

            img = img.to(device)
            img = torch.nn.functional.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
            features = model_feature(img)['avgpool'].squeeze(-1).squeeze(-1)
            assert features.shape[-1] == 2048 and features.ndim == 2
            features = features.detach().cpu().numpy()
            label = label.detach().cpu().numpy()

            with target_env.begin(write=True) as target_txn:
                for feature, lb in zip(features, label):
                    target_txn.put(f'feat-{str(idx)}'.encode('utf-8'), feature)
                    target_txn.put(f'y-{str(idx)}'.encode('utf-8'), str(lb).encode('utf-8'))
                    idx += 1

            if idx % 100 == 0:
                cur_time = time.time()
                print(f'saved {idx} files with {cur_time - begin}s elapsed')
                begin = time.time()

        if args.split == 'train':
            print('starting to store the xflip latents')
            begin = time.time()
            for batch in dataset_loader:
                img, label = batch
                assert img.min() >= -1 and img.max() <= 1

                img = img.to(device)
                img = torch.nn.functional.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
                features = model_feature(img.flip(dims=[-1]))['avgpool'].squeeze(-1).squeeze(-1)
                assert features.shape[-1] == 2048 and features.ndim == 2
                features = features.detach().cpu().numpy()
                label = label.detach().cpu().numpy()

                with target_env.begin(write=True) as target_txn:
                    for feature, lb in zip(features, label):
                        target_txn.put(f'feat-{str(idx)}'.encode('utf-8'), feature)
                        target_txn.put(f'y-{str(idx)}'.encode('utf-8'), str(lb).encode('utf-8'))
                        idx += 1

                if idx % 100 == 0:
                    cur_time = time.time()
                    print(f'saved {idx} files with {cur_time - begin}s elapsed')
                    begin = time.time()

        with target_env.begin(write=True) as target_txn:
            target_txn.put('length'.encode('utf-8'), str(idx).encode('utf-8'))

        print(f'[finished] saved {idx} files')

    extract_feature()


if __name__ == "__main__":
    main()