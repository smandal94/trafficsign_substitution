import warnings
warnings.filterwarnings("ignore")

import os, sys
import numpy as np
import cv2
import glob
import json
import pickle

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from einops.layers.torch import Rearrange
from torchvision.transforms import Compose
import pytorch_lightning as pl

import argparse
# import wandb
# from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from utils.utils import ToTensor, Normalize

from pl_cyclegan import CycleGANPL

# torch.cuda.set_device("cuda:0")
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
# cudnn.enabled = True
# cudnn.benckmark = True
# cudnn.deterministic = True
# device = torch.device('cuda:0')
# os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints', default='./checkpoints/')
parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
parser.add_argument('--netG', type=str, default='resnet_12blocks', help='specify generator architecture [resnet_12blocks | resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization [instance | batch | none]')
parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
parser.add_argument('--n_epochs_decay', type=int, default=500, help='number of epochs to linearly decay learning rate to zero')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
parser.add_argument('--lambda_identity', type=float, default=0.1, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
parser.add_argument('--lr_policy', type=str, default='cosine', help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
parser.add_argument('--dataroot', type=str, default='../extras/sign_data', help='path to dataset')
parser.add_argument('--momentsA', type=str, default='../extras/gtsrb128_inception_moments.npz', help='path to data A moments for FID')
parser.add_argument('--momentsB', type=str, default='../extras/B_inception_moments.npz', help='path to data A moments for FID')
# parser.add_argument('--dataroot', type=str, default='../cyclegan/datasets/horse2zebra_actual', help='path to dataset')
parser.add_argument('--img_size', type=int, default=64, help='image size')
parser.add_argument('--batch_size', type=int, default=64, help='image size')
parser.add_argument('--num_workers', type=int, default=32, help='image size')
parser.add_argument("--n_epochs", type=int, default=500, help='Number of epochs to run')
parser.add_argument("--resume_path", type=str, default=None, help='Checkpoint to resume')
parser.add_argument("--grad_accum", type=int, default=64, help='Gradient Accumulation to increase effective batchsize')
opt = parser.parse_args()

def main():
    if not os.path.exists(opt.output_directory):
        os.makedirs(opt.output_directory)
    
    ckpt_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        dirpath=opt.output_directory,
        filename="{epoch:02d}-{global_step}",
        monitor='global_step',
        verbose=True,
        save_top_k=3,
        mode='min',
        save_on_train_epoch_end=True
    )

    transform = Compose([
        ToTensor(),
        # Rearrange("h w c -> c h w"),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    opt.transform = transform
    opt.augmentation = None

    pl.utilities.seed.seed_everything(42)
    model = CycleGANPL(opt)
    print('Model: ', model)
    model.check_params()

    # wandb.login()
    # wandb_logger = WandbLogger(project="TrafficSign", group="CycleGAN_UNet128", config=opt)

    trainer = pl.Trainer(precision=32, 
                        # benchmark=True,
                        # gpus=-1,
                        accelerator='tpu',
                        devices=8,
                        # logger=wandb_logger,
                        logger=None,
                        max_epochs=opt.n_epochs,
                        enable_checkpointing=ckpt_callback,
                        deterministic=False,
                        track_grad_norm=True,
                        resume_from_checkpoint=opt.resume_path,
                        # accumulate_grad_batches=opt.grad_accum//opt.batch_size,
                        log_every_n_steps=1,
                        num_sanity_val_steps=0
                        )
    trainer.fit(model)

if __name__ == '__main__':
    main()
