import os, sys
from copy import deepcopy
import math
import numpy as np
import cv2
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter as P
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from models import networks
from utils.image_pool import ImagePool
from utils.utils import AverageMeter
from utils.dataset import UnalignedDataset
# from utils.horsedataset import UnalignedDataset
import itertools
import matplotlib.pyplot as plt
from copy import deepcopy

from models import inception_utils as iu

class CycleGANPL(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()

        assert(opt.input_nc == opt.output_nc)

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                    not opt.no_dropout, opt.init_type, opt.init_gain)
        
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                    not opt.no_dropout, opt.init_type, opt.init_gain)

        self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                    opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)
        
        self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                    opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)

        self.fake_A_pool = ImagePool(opt.pool_size)
        self.fake_B_pool = ImagePool(opt.pool_size)

        self.criterionGAN = networks.GANLoss(opt.gan_mode)
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()

        # losses accumulation
        # self.loss_idt_A
        self.tr_loss_idt_A = AverageMeter()
        self.tr_loss_idt_B = AverageMeter()
        self.tr_loss_G_A = AverageMeter()
        self.tr_loss_G_B = AverageMeter()
        self.tr_loss_cycle_A = AverageMeter()
        self.tr_loss_cycle_B = AverageMeter()
        self.tr_loss_G = AverageMeter()

        self.tr_loss_D = AverageMeter()
        self.tr_loss_D_A = AverageMeter()
        self.tr_loss_D_B = AverageMeter()

        self.opt = opt

        # FID & IS
        self.ISmodel = iu.load_inception_net()
        self.data_mu_A = np.load('gtsrb128_inception_moments.npz')['mu']
        self.data_sigma_A = np.load('gtsrb128_inception_moments.npz')['sigma']
        self.data_mu_B = np.load('B_inception_moments.npz')['mu']
        self.data_sigma_B = np.load('B_inception_moments.npz')['sigma']
        self.pool_A, self.logits_A = [], []
        self.pool_B, self.logits_B = [], []

        self.cache_A, self.cache_B = None, None

        # self.mean = P(torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1),
        #           requires_grad=False)
        # self.std = P(torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1),
        #             requires_grad=False)

        self.mean = 0.5
        self.std = 0.5

        # self.automatic_optimization = False
    
    def check_params(self):
        num_params = 0
        for param in self.netG_A.parameters():
            num_params += param.numel()
        print('[netG_A] Total number of parameters : %.3f M' % (num_params / 1e6))

        num_params = 0
        for param in self.netG_B.parameters():
            num_params += param.numel()
        print('[netG_B] Total number of parameters : %.3f M' % (num_params / 1e6))

        num_params = 0
        for param in self.netD_A.parameters():
            num_params += param.numel()
        print('[netD_A] Total number of parameters : %.3f M' % (num_params / 1e6))

        num_params = 0
        for param in self.netD_B.parameters():
            num_params += param.numel()
        print('[netD_B] Total number of parameters : %.3f M' % (num_params / 1e6))
    
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        d_opt = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

        sch1 = networks.get_scheduler(g_opt, self.opt)
        sch2 = networks.get_scheduler(d_opt, self.opt)

        return [g_opt, d_opt], [sch1, sch2]
    
    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # loss_D.backward()
        return loss_D
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        self.real_A = batch['A']
        self.real_B = batch['B']

        # one = self.real_A[3].cpu().numpy().transpose(1, 2, 0)
        # one = one * self.std + self.mean
        # one = one * 255.0
        # one = one.astype(np.uint8)
        # cv2.imwrite('trainA.jpg', one[:,:,::-1])

        # one = self.real_B[3].cpu().numpy().transpose(1, 2, 0)
        # one = one * self.std + self.mean
        # one = one * 255.0
        # one = one.astype(np.uint8)
        # cv2.imwrite('trainB.jpg', one[:,:,::-1])


        if self.cache_A is None:
            self.cache_A = deepcopy(self.real_A)
            self.cache_B = deepcopy(self.real_B)

        # g_opt, d_opt = self.optimizers()

        if optimizer_idx == 0:
            ######################
            # Optimize Generator #
            ######################

            self.fake_B = self.netG_A(deepcopy(self.real_A))                # G_A(A)
            self.rec_A = self.netG_B(self.fake_B)                 # G_B(G_A(A))
            self.fake_A = self.netG_B(deepcopy(self.real_B))                # G_B(B)
            self.rec_B = self.netG_A(self.fake_A)                 # G_A(G_B(B))

            self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gsz
            # g_opt.zero_grad()  # set G_A and G_B's gradients to zero

            lambda_idt = self.opt.lambda_identity
            lambda_A = self.opt.lambda_A
            lambda_B = self.opt.lambda_B
            # Identity loss
            if lambda_idt > 0:
                self.idt_A = self.netG_A(deepcopy(self.real_B))
                loss_idt_A = self.criterionIdt(self.idt_A, deepcopy(self.real_B)) * lambda_B * lambda_idt
                self.idt_B = self.netG_B(deepcopy(self.real_A))
                loss_idt_B = self.criterionIdt(self.idt_B, deepcopy(self.real_A)) * lambda_A * lambda_idt
            else:
                loss_idt_A = torch.tensor(0.0)
                loss_idt_B = torch.tensor(0.0)

            # GAN loss D_A(G_A(A))
            loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
            # GAN loss D_B(G_B(B))
            loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
            # Forward cycle loss || G_B(G_A(A)) - A||
            loss_cycle_A = self.criterionCycle(self.rec_A, deepcopy(self.real_A)) * lambda_A
            # Backward cycle loss || G_A(G_B(B)) - B||
            loss_cycle_B = self.criterionCycle(self.rec_B, deepcopy(self.real_B)) * lambda_B
            # combined loss and calculate gradients
            loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B

            # self.manual_backward(loss_G)
            # g_opt.step()       # update G_A and G_B's weights

            self.tr_loss_idt_A.update(loss_idt_A.item(), len(self.real_A))
            self.tr_loss_idt_B.update(loss_idt_B.item(), len(self.real_A))
            self.tr_loss_G_A.update(loss_G_A.item(), len(self.real_A))
            self.tr_loss_G_B.update(loss_G_B.item(), len(self.real_A))
            self.tr_loss_cycle_A.update(loss_cycle_A.item(), len(self.real_A))
            self.tr_loss_cycle_B.update(loss_cycle_B.item(), len(self.real_A))
            self.tr_loss_G.update(loss_G.item(), len(self.real_A))

            self.log('train/loss_G_step', self.tr_loss_G.avg, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log('train/loss_idt_A_step', self.tr_loss_idt_A.avg, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log('train/loss_idt_B_step', self.tr_loss_idt_B.avg, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log('train/loss_G_A_step', self.tr_loss_G_A.avg, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log('train/loss_G_B_step', self.tr_loss_G_B.avg, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            return {'loss': loss_G}
        else:
            ##########################
            # Optimize Discriminator #
            ##########################

            self.set_requires_grad([self.netD_A, self.netD_B], True)
            # d_opt.zero_grad()   # set D_A and D_B's gradients to zero

            # calculate gradients for D_A
            """Calculate GAN loss for discriminator D_A"""
            self.fake_B = self.fake_B_pool.query(self.fake_B)
            loss_D_A = self.backward_D_basic(self.netD_A, deepcopy(self.real_B), self.fake_B)

            # calculate graidents for D_B
            """Calculate GAN loss for discriminator D_B"""
            self.fake_A = self.fake_A_pool.query(self.fake_A)
            loss_D_B = self.backward_D_basic(self.netD_B, deepcopy(self.real_A), self.fake_A)

            # d_opt.step()  # update D_A and D_B's weights
            loss_D = loss_D_A + loss_D_B

            self.tr_loss_D.update(loss_D.item(), len(self.real_A))
            self.tr_loss_D_A.update(loss_D_A.item(), len(self.real_A))
            self.tr_loss_D_B.update(loss_D_B.item(), len(self.real_A))

            self.log('train/loss_D_step', self.tr_loss_D.avg, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log('train/loss_D_A_step', self.tr_loss_D_A.avg, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log('train/loss_D_B_step', self.tr_loss_D_B.avg, on_step=True, on_epoch=False, prog_bar=False, logger=True)

            return {'loss': loss_D}
    
    def training_epoch_end(self, outputs):
        avg_G_loss = self.tr_loss_G.avg
        avg_D_loss = self.tr_loss_D.avg

        self.log('train/loss_G', avg_G_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('train/loss_D', avg_D_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.tr_loss_idt_A.reset()
        self.tr_loss_idt_B.reset()
        self.tr_loss_G_A.reset()
        self.tr_loss_G_B.reset()
        self.tr_loss_cycle_A.reset()
        self.tr_loss_cycle_B.reset()
        self.tr_loss_G.reset()

        self.tr_loss_D.reset()
        self.tr_loss_D_A.reset()
        self.tr_loss_D_B.reset()
    
    def validation_step(self, batch, batch_idx):
        self.real_A = batch['A']
        self.real_B = batch['B']

        self.fake_B = self.netG_A(deepcopy(self.real_A))                # G_A(A)
        self.fake_B = self.fake_B * self.std + self.mean
        pool_val, logits_val = self.ISmodel(self.fake_B)
        self.pool_B += [pool_val]
        self.logits_B += [F.softmax(logits_val, 1)]

        self.fake_A = self.netG_B(deepcopy(self.real_B))                # G_B(B)
        self.fake_A = self.fake_A * self.std + self.mean
        pool_val, logits_val = self.ISmodel(self.fake_A)
        self.pool_A += [pool_val]
        self.logits_A += [F.softmax(logits_val, 1)]

        IS_mean, IS_std = iu.calculate_inception_score(F.softmax(logits_val, 1).cpu().numpy())
        return {'val_IS_mean': IS_mean, 'val_IS_std': IS_std}
    
    def validation_epoch_end(self, outputs):
        self.pool_A = torch.cat(self.pool_A, 0)
        self.logits_A = torch.cat(self.logits_A, 0)
        # IS_mean, IS_std = iu.calculate_inception_score(self.logits_A.cpu().numpy())
        mu, sigma = torch.mean(self.pool_A, 0), iu.torch_cov(self.pool_A, rowvar=False)
        FID_A = iu.numpy_calculate_frechet_distance(mu.cpu().numpy(), sigma.cpu().numpy(), self.data_mu_A, self.data_sigma_A)
        # print('fida: ', FID_A)
        if not np.isnan(FID_A):
            self.log('val/FID_A', FID_A, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.pool_B = torch.cat(self.pool_B, 0)
        self.logits_B = torch.cat(self.logits_B, 0)
        # IS_mean, IS_std = iu.calculate_inception_score(self.logits_A.cpu().numpy())
        mu, sigma = torch.mean(self.pool_B, 0), iu.torch_cov(self.pool_B, rowvar=False)
        FID_B = iu.numpy_calculate_frechet_distance(mu.cpu().numpy(), sigma.cpu().numpy(), self.data_mu_B, self.data_sigma_B)

        # print('fidb: ', FID_B)
        if not np.isnan(FID_B):
            self.log('val/FID_B', FID_B, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.fake_B = self.netG_A(self.cache_A)                # G_A(A)
        self.fake_B = torch.cat([self.cache_A, self.fake_B], dim=0)
        self.fake_B = self.fake_B * self.std + self.mean
        self.fake_B = self.fake_B * 255.0
        joined_images_tensor_B = make_grid(self.fake_B, nrow=self.opt.batch_size//2, padding=2)
        joined_images_B = joined_images_tensor_B.detach().cpu().numpy().astype(np.uint8)
        joined_images_B = joined_images_B.transpose(1, 2, 0)
        # cv2.imwrite('A2B.jpg', joined_images_B[:,:,::-1])

        # one = self.cache_A[1].cpu().numpy().transpose(1, 2, 0)
        # one = one * self.std + self.mean
        # one = one * 255.0
        # one = one.astype(np.uint8)
        # cv2.imwrite('valA.jpg', one[:,:,::-1])

        # one = self.cache_B[1].cpu().numpy().transpose(1, 2, 0)
        # one = one * self.std + self.mean
        # one = one * 255.0
        # one = one.astype(np.uint8)
        # cv2.imwrite('valB.jpg', one[:,:,::-1])

        # Visualize
        fig = plt.figure(figsize=(18, 8))
        plt.imshow(joined_images_B)
        plt.axis('off')
        plt.title(f'Epoch {self.current_epoch} A->B')
        plt.savefig('train_figs/A2B/A2B.png', bbox_inches='tight')

        self.fake_A = self.netG_B(self.cache_B)                # G_B(B)
        self.fake_A = torch.cat([self.cache_B, self.fake_A], dim=0)
        self.fake_A = self.fake_A * self.std + self.mean
        self.fake_A = self.fake_A * 255.0
        joined_images_tensor_A = make_grid(self.fake_A, nrow=self.opt.batch_size//2, padding=2)
        joined_images_A = joined_images_tensor_A.detach().cpu().numpy().astype(np.uint8)
        # joined_images_A = np.transpose(joined_images_A, [1,2,0])
        joined_images_A = joined_images_A.transpose(1, 2, 0)
        # cv2.imwrite('B2A.jpg', joined_images_A[:,:,::-1])

        # Visualize
        fig = plt.figure(figsize=(18, 8))
        plt.imshow(joined_images_A)
        plt.axis('off')
        plt.title(f'Epoch {self.current_epoch} B->A')
        plt.savefig('train_figs/B2A/B2A.png', bbox_inches='tight')

        # self.log('A->B', joined_images_B, on_step=False, on_epoch=True, prog_bar=False, logger=True) 
        # self.log('B->A', joined_images_A, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # self.logger.experiment.log({"examples": [wandb.Image(joined_images_B, caption="Epoch: {}, A->B".format(self.current_epoch)), wandb.Image(joined_images_A, caption="Epoch: {}, B->A".format(self.current_epoch))]}, commit=False)
        # wandb_logger.log_image(key="samples", images=[joined_images_B, joined_images_A], caption=["Epoch: {}, A->B".format(self.current_epoch), "Epoch: {}, B->A".format(self.current_epoch)])

        self.cache_A = None
        self.cache_B = None

        self.logits_A = []
        self.pool_A = []
        self.logits_B = []
        self.pool_B = []

        return {'val_FID_A': FID_A, 'val_FID_B': FID_B}



    def train_dataloader(self):
        self.traindataset = UnalignedDataset(self.opt, transform=self.opt.transform, augmentation=self.opt.augmentation)
        return DataLoader(self.traindataset, batch_size=self.opt.batch_size, num_workers=self.opt.num_workers, shuffle=True, pin_memory=False, drop_last=False, sampler=None, collate_fn=self.traindataset.collate)

    def val_dataloader(self):
        self.valdataset = UnalignedDataset(self.opt, transform=self.opt.transform, augmentation=self.opt.augmentation)
        return DataLoader(self.valdataset, batch_size=self.opt.batch_size, num_workers=self.opt.num_workers, shuffle=False, pin_memory=False, drop_last=False, sampler=None, collate_fn=self.valdataset.collate)

def main():
    print('Hello World')

if __name__ == '__main__':
    main()