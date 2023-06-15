# MIT License

# Copyright (c) [2023] [Anima-Lab]


import io
import os
import json
import zipfile

import lmdb
import numpy as np
from PIL import Image
import torch
from torchvision.datasets import ImageFolder, VisionDataset



def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
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
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


################################################################################
# ImageNet - LMDB
###############################################################################

def lmdb_loader(path, lmdb_data, resolution):
    # In-memory binary streams
    with lmdb_data.begin(write=False, buffers=True) as txn:
        bytedata = txn.get(path.encode('ascii'))
    img = Image.open(io.BytesIO(bytedata)).convert('RGB')
    arr = center_crop_arr(img, resolution)
    # arr = arr.astype(np.float32) / 127.5 - 1
    # arr = np.transpose(arr, [2, 0, 1])  # CHW
    return arr


def imagenet_lmdb_dataset(
        root, 
        transform=None, target_transform=None, 
        resolution=256):
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

    lmdb_dataset = ImageLMDB(lmdb_path, transform, target_transform, resolution, data_set.imgs, data_set.class_to_idx, data_set.classes)
    return lmdb_dataset


################################################################################
# ImageNet Dataset class- LMDB
###############################################################################

class ImageLMDB(VisionDataset):
    """
    A data loader for ImageNet LMDB dataset, which is faster than the original ImageFolder.
    """
    def __init__(self, root, transform=None, target_transform=None, 
                 resolution=256, samples=None, class_to_idx=None, classes=None):
        super().__init__(root, transform=transform,
                         target_transform=target_transform)
        self.root = root
        self.resolution = resolution
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.classes = classes
    
    def __getitem__(self, index: int):
        path, target = self.samples[index]

        # load image from path
        if not hasattr(self, 'txn'):
            self.open_db()
        bytedata = self.txn.get(path.encode('ascii'))
        img = Image.open(io.BytesIO(bytedata)).convert('RGB')
        arr = center_crop_arr(img, self.resolution)
        if self.transform is not None:
            arr = self.transform(arr)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return arr, target

    def __len__(self) -> int:
        return len(self.samples)

    def open_db(self):
        self.env = lmdb.open(self.root, readonly=True, max_readers=256, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False, buffers=True)



################################################################################
# ImageNet - LMDB - latent space
###############################################################################



# ----------------------------------------------------------------------------
# Abstract base class for datasets.

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 name,  # Name of the dataset.
                 raw_shape,  # Shape of the raw image data (NCHW).
                 max_size=None,  # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
                 label_dim=1000,  # Ensure specific number of classes
                 xflip=False,  # Artificially double the size of the dataset via x-flips. Applied after max_size.
                 random_seed=0,  # Random seed to use when applying max_size.
                 ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._label_dim = label_dim
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip. (Assume the dataset already contains the same number of xflipped samples)
        if xflip:
            self._raw_idx = np.concatenate([self._raw_idx, self._raw_idx + self._raw_shape[0]])

    def close(self):  # to be overridden by subclass
        pass

    def _load_raw_data(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image, cond = self._load_raw_data(raw_idx)
        assert isinstance(image, np.ndarray)
        if isinstance(cond, list):  # [label, feature]
            cond[0] = self._get_onehot(cond[0])
        else:  # label
            cond = self._get_onehot(cond)
        return image.copy(), cond

    def _get_onehot(self, label):
        if isinstance(label, int) or label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        assert isinstance(label, np.ndarray)
        return label.copy()

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            self._label_shape = [self._label_dim]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return True


# ----------------------------------------------------------------------------
# Dataset subclass that loads latent images recursively from the specified lmdb file.

class ImageNetLatentDataset(Dataset):
    def __init__(self,
                 path,  # Path to directory or zip.
                 resolution=32,  # Ensure specific resolution, default 32.
                 num_channels=4,  # Ensure specific number of channels, default 4.
                 split='train',  # train or val split
                 feat_path=None, # Path to features lmdb file (only works when feat_cond=True)
                 feat_dim=0,  # feature dim
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):
        self._path = os.path.join(path, split)
        self.feat_dim = feat_dim
        if not hasattr(self, 'txn'):
            self.open_lmdb()
        self.feat_txn = None
        if feat_path is not None and os.path.isdir(feat_path):
            assert self.feat_dim > 0
            self._feat_path = os.path.join(feat_path, split)
            self.open_feat_lmdb()

        length = int(self.txn.get('length'.encode('utf-8')).decode('utf-8'))
        name = os.path.basename(path)
        raw_shape = [length // 2, num_channels, resolution, resolution]  # 1281167 x 4 x 32 x 32
        if raw_shape[2] != resolution or raw_shape[3] != resolution:
            raise IOError('Image files do not match the specified resolution')

        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    def open_lmdb(self):
        self.env = lmdb.open(self._path, readonly=True, lock=False, create=False)
        self.txn = self.env.begin(write=False)

    def open_feat_lmdb(self):
        self.feat_env = lmdb.open(self._feat_path, readonly=True, lock=False, create=False)
        self.feat_txn = self.feat_env.begin(write=False)

    def _load_raw_data(self, idx):
        if not hasattr(self, 'txn'):
            self.open_lmdb()

        z_bytes = self.txn.get(f'z-{str(idx)}'.encode('utf-8'))
        y_bytes = self.txn.get(f'y-{str(idx)}'.encode('utf-8'))
        z = np.frombuffer(z_bytes, dtype=np.float32).reshape([-1, self.resolution, self.resolution]).copy()
        y = int(y_bytes.decode('utf-8'))

        cond = y
        if self.feat_txn is not None:
            feat_bytes = self.feat_txn.get(f'feat-{str(idx)}'.encode('utf-8'))
            feat_y_bytes = self.feat_txn.get(f'y-{str(idx)}'.encode('utf-8'))
            feat = np.frombuffer(feat_bytes, dtype=np.float32).reshape([self.feat_dim]).copy()
            feat_y = int(feat_y_bytes.decode('utf-8'))
            assert y == feat_y, 'Ordering mismatch between txn and feat_txn!'
            cond = [y, feat]

        return z, cond

    def close(self):
        try:
            if self.env is not None:
                self.env.close()
            if self.feat_env is not None:
                self.feat_env.close()
        finally:
            self.env = None
            self.feat_env = None


# ----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory or zip file.

class ImageFolderDataset(Dataset):
    def __init__(self,
                 path,  # Path to directory or zip.
                 resolution=None,  # Ensure specific resolution, None = highest available.
                 use_labels=False, # Enable conditioning labels? False = label dimension is zero.
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):
        self._path = path
        self._zipfile = None
        self._raw_labels = None
        self._use_labels = use_labels

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in
                                os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_data(self, raw_idx):
        image = self._load_raw_image(raw_idx)
        assert image.dtype == np.uint8
        label = self._get_raw_labels()[raw_idx]
        return image, label

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            image = np.array(Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

# ----------------------------------------------------------------------------
