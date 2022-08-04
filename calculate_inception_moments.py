import os, sys

# remove later
sys.path.append('../')

import numpy as np
import cv2
import subprocess
import glob
import random
from tqdm import tqdm

import torch
import models.inception_utils as iu
from utils.dataset import UnalignedDataset

from einops.layers.torch import Rearrange
from utils.utils import ToTensor, Normalize
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
import torch.nn.functional as F

cv2.setNumThreads(1)

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='../trafficsign_substitution_temp/sign_data', help='path to dataset')
    parser.add_argument('--img_size', type=int, default=128, help='image size')
    opt = parser.parse_args()

    model = iu.load_inception_net()
    model.cuda()
    print(model)

    transform = Compose([
        ToTensor(),
        Rearrange("h w c -> c h w")
    ])

    x = UnalignedDataset(opt, transform=transform)
    data = DataLoader(x, batch_size=32, num_workers=8, shuffle=False, pin_memory=False, drop_last=False, sampler=None, collate_fn=x.collate)

    pool, logits, labels = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data)):
            x = batch['A']
            y = batch['A_labels']
            x = x.cuda()
            pool_val, logits_val = model(x)
            pool += [np.asarray(pool_val.cpu())]
            logits += [np.asarray(F.softmax(logits_val, 1).cpu())]
            labels += [np.asarray(y.cpu())]

    pool, logits, labels = [np.concatenate(item, 0) for item in [pool, logits, labels]]
    
    print('Calculating inception metrics...')
    IS_mean, IS_std = iu.calculate_inception_score(logits)
    print('Training data from dataset A has IS of %5.5f +/- %5.5f' % (IS_mean, IS_std))
    print('Calculating means and covariances...')
    mu, sigma = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
    print('Saving calculated means and covariances to disk...')
    np.savez('gtsrb128_inception_moments.npz', **{'mu' : mu, 'sigma' : sigma})

    pool, logits, labels = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data)):
            x = batch['B']
            y = batch['A_labels']
            x = x.cuda()
            pool_val, logits_val = model(x)
            pool += [np.asarray(pool_val.cpu())]
            logits += [np.asarray(F.softmax(logits_val, 1).cpu())]
            labels += [np.asarray(y.cpu())]

    pool, logits, labels = [np.concatenate(item, 0) for item in [pool, logits, labels]]
    
    print('Calculating inception metrics...')
    IS_mean, IS_std = iu.calculate_inception_score(logits)
    print('Training data from dataset B has IS of %5.5f +/- %5.5f' % (IS_mean, IS_std))
    print('Calculating means and covariances...')
    mu, sigma = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
    print('Saving calculated means and covariances to disk...')
    np.savez('B_inception_moments.npz', **{'mu' : mu, 'sigma' : sigma})

    FID = iu.numpy_calculate_frechet_distance(mu, sigma, mu, sigma)
    print('FID: {}'.format(FID))

if __name__ == '__main__':
    main()