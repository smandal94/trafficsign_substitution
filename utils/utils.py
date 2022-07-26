import os, sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import random
from statistics import mean
from tqdm import tqdm
import subprocess
import cv2
from copy import deepcopy
import numbers

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ToTensor:
    def __call__(self, x):
        return torch.from_numpy(np.array(x)).float() / 255.

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, video):
        shape = (-1,) + (1,) * (video.dim() - 1)

        mean = torch.as_tensor(self.mean, device=video.device).reshape(shape)
        std = torch.as_tensor(self.std, device=video.device).reshape(shape)

        return (video - mean) / std