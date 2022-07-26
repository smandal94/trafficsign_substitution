import os, sys
import numpy as np
import cv2
import subprocess
import glob
import random

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

from einops.layers.torch import Rearrange
from utils.utils import ToTensor, Normalize
from torchvision.transforms import Compose
from torch.utils.data import DataLoader

cv2.setNumThreads(1)

class UnalignedDataset(Dataset):
    def __init__(self, opt, phase='train', transform=None, augmentation=None):
        self.augmentation = augmentation
        self.transform = transform
        self.opt = opt

        assert transform is not None
        
        self.dir_A = os.path.join(opt.dataroot, phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, phase + 'B')
        # print('A: ', self.dir_A)
        # print('B: ', self.dir_B)

        self.A_paths = sorted(glob.glob(self.dir_A + '/*/*.ppm'))
        self.B_paths = sorted(glob.glob(self.dir_B + '/*.png'))
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        '''
        ###############
        # For testing #
        ###############

        for i in range(self.A_size):
            img = cv2.imread(self.A_paths[i], 1)
            img = cv2.resize(img, (self.opt.img_size, self.opt.img_size), None, interpolation=cv2.INTER_CUBIC)
            print('\n',self.A_paths[i])
            cv2.imshow('img', img)
            key = cv2.waitKey(0)
            if key == 27:
                cv2.destroyAllWindows()
                exit()
        
        for i in range(self.B_size):
            img = cv2.imread(self.B_paths[i], -1)
            img = cv2.resize(img, (self.opt.img_size, self.opt.img_size), None, interpolation=cv2.INTER_CUBIC)
            fg = img[:,:,:-1]
            alpha = cv2.cvtColor(img[:,:,-1], cv2.COLOR_GRAY2RGB)
            print(img.shape)
            print('\n',self.B_paths[i])

            color = tuple(np.random.choice(range(255),size=3))
            bg = np.zeros((self.opt.img_size, self.opt.img_size, 3), np.uint8)
            bg[:] = color

            fg = fg.astype(float) / 255.0
            bg = bg.astype(float) / 255.0
            alpha = alpha.astype(float) / 255.0
            fg = cv2.multiply(alpha, fg)
            bg = cv2.multiply(1.0 - alpha, bg)
            img = cv2.add(fg, bg)
            img = img * 255.0
            img = img.astype(np.uint8)

            cv2.imshow('img', img)
            key = cv2.waitKey(0)
            if key == 27:
                cv2.destroyAllWindows()
                exit()
        '''
        
    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    def __getitem__(self, idx):
        A_path = self.A_paths[idx % self.A_size]  # make sure index is within then range
        idx_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[idx_B]

        A_img = cv2.imread(A_path, 1)
        A_img = cv2.resize(A_img, (self.opt.img_size, self.opt.img_size), None, interpolation=cv2.INTER_CUBIC)

        B_img = cv2.imread(B_path, -1)
        B_img = cv2.resize(B_img, (self.opt.img_size, self.opt.img_size), None, interpolation=cv2.INTER_CUBIC)
        fg = B_img[:,:,:-1]
        alpha = cv2.cvtColor(B_img[:,:,-1], cv2.COLOR_GRAY2RGB)
        color = tuple(np.random.choice(range(255),size=3))
        bg = np.zeros((self.opt.img_size, self.opt.img_size, 3), np.uint8)
        bg[:] = color
        fg = fg.astype(float) / 255.0
        bg = bg.astype(float) / 255.0
        alpha = alpha.astype(float) / 255.0
        fg = cv2.multiply(alpha, fg)
        bg = cv2.multiply(1.0 - alpha, bg)
        B_img = cv2.add(fg, bg)
        B_img = B_img * 255.0
        B_img = B_img.astype(np.uint8)

        if self.augmentation is not None:
            chance = random.randint(0, 4)
            if chance == 1:
              A_img = self.augmentation(A_img)
        A_img = np.array(A_img, dtype=np.uint8)

        A = self.transform(A_img)
        B = self.transform(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
    
    def collate(self, batch):
        input_A = [b['A'] for b in batch]
        input_A = torch.stack(input_A)

        input_B = [b['B'] for b in batch]
        input_B = torch.stack(input_B)

        A_paths = [b['A_paths'] for b in batch]
        B_paths = [b['B_paths'] for b in batch]

        return {'A': input_A, 'B': input_B, 'A_paths': A_paths, 'B_paths': B_paths}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='../sign_data', help='path to dataset')
    parser.add_argument('--img_size', type=int, default=128, help='image size')
    opt = parser.parse_args()

    transform = Compose([
        ToTensor(),
        Rearrange("h w c -> c h w"),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    x = UnalignedDataset(opt, transform=transform)
    data = DataLoader(x, batch_size=8, num_workers=8, shuffle=True, pin_memory=False, drop_last=False, sampler=None, collate_fn=x.collate)

    for i, x in enumerate(data):
        A = x['A']
        B = x['B']
        print('\nA: ', A.shape)
        print('B: ', B.shape)
        print(x['A_paths'])

if __name__ == '__main__':
    main()