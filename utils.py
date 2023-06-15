# MIT License

# Copyright (c) [2023] [Anima-Lab]

# This code is adapted from https://github.com/NVlabs/edm/blob/main/generate.py. 
# The original code is licensed under a Creative Commons 
# Attribution-NonCommercial-ShareAlike 4.0 International License, which is can be found at licenses/LICENSE_EDM.txt. 


import os
import re
import sys
import contextlib

import torch
import torch.distributed as dist


#----------------------------------------------------------------------------
# Get the latest checkpoint from the save dir

def get_latest_ckpt(dir):
    latest_id = -1
    for file in os.listdir(dir):
        if file.endswith('.pt'):
            m = re.search(r'(\d+)\.pt', file)
            if m:
                ckpt_id = int(m.group(1))
                latest_id = max(latest_id, ckpt_id)
    if latest_id == -1:
        return None
    else:
        ckpt_path = os.path.join(dir, f'{latest_id:07d}.pt')
        return ckpt_path


def get_ckpt_paths(dir, id_min, id_max):
    ckpt_dict = {}
    for file in os.listdir(dir):
        if file.endswith('.pt'):
            m = re.search(r'(\d+)\.pt', file)
            if m:
                ckpt_id = int(m.group(1))
                if id_min <= ckpt_id <= id_max:
                    ckpt_dict[ckpt_id] = os.path.join(dir, f'{ckpt_id:07d}.pt')
    return ckpt_dict


#----------------------------------------------------------------------------
# Take the mean over all non-batch dimensions.

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, tensor.ndim)))


#----------------------------------------------------------------------------
# Convert latent (mean, logvar) to latent variable (inherited from autoencoder.py)

def sample(moments, scale_factor=0.18215):
    mean, logvar = torch.chunk(moments, 2, dim=1)
    logvar = torch.clamp(logvar, -30.0, 20.0)
    std = torch.exp(0.5 * logvar)
    z = mean + std * torch.randn_like(mean)
    z = scale_factor * z
    return z


#----------------------------------------------------------------------------
# Context manager for easily enabling/disabling DistributedDataParallel
# synchronization.

@contextlib.contextmanager
def ddp_sync(module, sync):
    assert isinstance(module, torch.nn.Module)
    if sync or not isinstance(module, torch.nn.parallel.DistributedDataParallel):
        yield
    else:
        with module.no_sync():
            yield

#----------------------------------------------------------------------------
#  Distributed training helper functions

def init_processes(fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6020'
    print(f'MASTER_ADDR = {os.environ["MASTER_ADDR"]}')
    print(f'MASTER_PORT = {os.environ["MASTER_PORT"]}')
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=args.global_rank, world_size=args.global_size)
    fn(args)
    if args.global_size > 1:
        cleanup()


def mprint(*args, **kwargs):
    """
    Print only from rank 0.
    """
    if dist.get_rank() == 0:
        print(*args, **kwargs)


def cleanup():
    """
    End DDP training.
    """
    dist.barrier()
    mprint("Done!")
    dist.barrier()
    dist.destroy_process_group()


#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

# Parse 'None' to None and others to float value
def parse_float_none(s):
    assert isinstance(s, str)
    return None if s == 'None' else float(s)

# Parse 'None' to None and others to str
def parse_str_none(s):
    assert isinstance(s, str)
    return None if s == 'None' else s

# Parse 'true' to True
def str2bool(s):
    return s.lower() in ['true', '1', 'yes']


#----------------------------------------------------------------------------
# logging info.
class Logger(object):
    """
    Redirect stderr to stdout, optionally print stdout to a file,
    and optionally force flushing on both stdout and the file.
    """

    def __init__(self, file_name=None, file_mode="w", should_flush=True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write(self, text):
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self):
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self):
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()