import numpy as np
from scipy.interpolate import splrep, splev
import torch
import math


def time_aug(X, t_range=(0, 1)):
    # assume (batch, time, channel)
    t_len = X.shape[1]
    time_axis = torch.linspace(*t_range, t_len, device=X.device)
    time_component = torch.repeat_interleave(time_axis[None, :, None], len(X), dim=0)
    return torch.cat([time_component, X], dim=-1)


def znorm(x, dim=-1, eps=1e-8):
    if torch.is_tensor(x):
        return (x - x.mean(dim, keepdim=True)) / (eps + x.std(dim, keepdim=True))
    else:
        return (x - x.mean(dim, keepdims=True)) / (eps + x.std(dim, keepdims=True))


def minmaxnorm(x, dim=-1, eps=1e-8):
    if torch.is_tensor(x):
        return (x - x.amin(dim, keepdim=True)) / (
            eps + x.amax(dim, keepdim=True) - x.amin(dim, keepdim=True)
        )
    else:
        return (x - x.amin(dim, keepdims=True)) / (
            eps + x.amax(dim, keepdims=True) - x.amin(dim, keepdims=True)
        )


def unorm(x, dim=-1, eps=1e-8):
    return x / (eps + torch.norm(x, p=2, dim=dim, keepdim=True))


def max_abs_norm(x, dim=-1, eps=1e-8):
    return x / (eps + torch.max(torch.abs(x), dim=dim, keepdim=True).values)


def to_leadlag(X):
    if not torch.is_tensor(X):
        X = torch.tensor(X).float()
    # [batch, time, (channel)]
    X_repeat = X.repeat_interleave(2, dim=1)

    # Split out lead and lag
    lead = X_repeat[:, 1:, :]
    lag = X_repeat[:, :-1, :]

    # Combine
    X_leadlag = torch.cat((lead, lag), 2)

    return X_leadlag


def resample(X, resample, smooth=0, k=3):
    # TODO: batch optimisation?
    spls = [
        splrep(np.linspace(0, 1, X.shape[-1]), X[i, :], k=k, s=smooth)
        for i in range(X.shape[0])
    ]
    X_rs = [splev(np.linspace(0, 1, resample), spl) for spl in spls]
    return np.stack(X_rs, axis=0)


# NOTE: borrowed
def rescale_path(path, depth):
    coeff = math.factorial(depth) ** (1 / depth)
    return coeff * path
