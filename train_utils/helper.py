# MIT License

# Copyright (c) [2023] [Anima-Lab]
from collections import OrderedDict
import torch
import numpy as np


def get_mask_ratio_fn(name='constant', ratio_scale=0.5, ratio_min=0.0):
    if name == 'cosine2':
        return lambda x: (ratio_scale - ratio_min) * np.cos(np.pi * x / 2) ** 2 + ratio_min
    elif name == 'cosine3':
        return lambda x: (ratio_scale - ratio_min) * np.cos(np.pi * x / 2) ** 3 + ratio_min
    elif name == 'cosine4':
        return lambda x: (ratio_scale - ratio_min) * np.cos(np.pi * x / 2) ** 4 + ratio_min
    elif name == 'cosine5':
        return lambda x: (ratio_scale - ratio_min) * np.cos(np.pi * x / 2) ** 5 + ratio_min
    elif name == 'cosine6':
        return lambda x: (ratio_scale - ratio_min) * np.cos(np.pi * x / 2) ** 6 + ratio_min
    elif name == 'exp':
        return lambda x: (ratio_scale - ratio_min) * np.exp(-x * 7) + ratio_min
    elif name == 'linear':
        return lambda x: (ratio_scale - ratio_min) * x + ratio_min
    elif name == 'constant':
        return lambda x: ratio_scale
    else:
        raise ValueError('Unknown mask ratio function: {}'.format(name))
    

def get_one_hot(labels, num_classes=1000):
    one_hot = torch.zeros(labels.shape[0], num_classes, device=labels.device)
    one_hot.scatter_(1, labels.view(-1, 1), 1)
    return one_hot


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


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
        if param.requires_grad:
            ema_name = name.replace('_orig_mod.', '')
            ema_params[ema_name].mul_(decay).add_(param.data, alpha=1 - decay)


def unwrap_model(model):
    """
    Unwrap a model from any distributed or compiled wrappers. 
    """
    if isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
        model = model._orig_mod
    if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
        model = model.module
    return model