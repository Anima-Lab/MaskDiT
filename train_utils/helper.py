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