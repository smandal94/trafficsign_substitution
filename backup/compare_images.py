import os, sys
import numpy as np
import cv2
import glob
from tqdm import tqdm
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from statistics import mean

import lpips

np.seterr(all='print')

def main1():
    files = sorted(glob.glob('/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/results/DFG_resultcrops/iter1/real/*.png'))
    print('No. of files: ', len(files))

    mse_list, ssim_list, psnr_list, lpips_list = [], [], [], []

    loss_fn = lpips.LPIPS(net='alex',version='0.1')
    loss_fn.cuda()

    for fR in tqdm(files):
        imgR = cv2.imread(fR, 1)
        fS = fR.replace('real', 'synthetic')
        imgS = cv2.imread(fS, 1)
        assert imgR.shape == imgS.shape

        if imgR.shape[0] < 7 or imgR.shape[1] < 7:
            continue

        mse = mean_squared_error(imgR, imgS)
        if mse == 0.0:
            # cv2.imshow('img', np.hstack((imgR, imgS)))
            # key = cv2.waitKey(0)
            # if key == 27:
            #     cv2.destroyAllWindows()
            #     exit()
            continue
        ssim = structural_similarity(imgR, imgS, channel_axis=2)
        psnr = peak_signal_noise_ratio(imgR, imgS)

        imgR = cv2.resize(imgR, (64, 64), None, interpolation=cv2.INTER_AREA)
        imgS = cv2.resize(imgS, (64, 64), None, interpolation=cv2.INTER_AREA)

        img0 = lpips.im2tensor(imgR[:,:,::-1])
        img1 = lpips.im2tensor(imgS[:,:,::-1])
        img0 = img0.cuda()
        img1 = img1.cuda()
        dist01 = loss_fn.forward(img0,img1).item()
        
        mse_list.append(mse)
        ssim_list.append(ssim)
        psnr_list.append(psnr)
        lpips_list.append(dist01)

    print('Iteration1 :: MSE: {:.02f}, SSIM: {:.02f}, PSNR: {:.02f}, LPIPS: {:.02f}'.format(mean(mse_list), mean(ssim_list), mean(psnr_list), mean(lpips_list)))

def main2():
    files = sorted(glob.glob('/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/results/DFG_resultcrops/iter2/real/*.png'))
    print('No. of files: ', len(files))

    mse_list, ssim_list, psnr_list, lpips_list = [], [], [], []

    loss_fn = lpips.LPIPS(net='alex',version='0.1')
    loss_fn.cuda()

    for fR in tqdm(files):
        imgR = cv2.imread(fR, 1)
        fS = fR.replace('real', 'synthetic_poisson')
        imgS = cv2.imread(fS, 1)
        assert imgR.shape == imgS.shape

        if imgR.shape[0] < 7 or imgR.shape[1] < 7:
            continue

        mse = mean_squared_error(imgR, imgS)
        if mse == 0.0:
            # cv2.imshow('img', np.hstack((imgR, imgS)))
            # key = cv2.waitKey(0)
            # if key == 27:
            #     cv2.destroyAllWindows()
            #     exit()
            continue
        ssim = structural_similarity(imgR, imgS, channel_axis=2)
        psnr = peak_signal_noise_ratio(imgR, imgS)

        imgR = cv2.resize(imgR, (64, 64), None, interpolation=cv2.INTER_AREA)
        imgS = cv2.resize(imgS, (64, 64), None, interpolation=cv2.INTER_AREA)

        img0 = lpips.im2tensor(imgR[:,:,::-1])
        img1 = lpips.im2tensor(imgS[:,:,::-1])
        img0 = img0.cuda()
        img1 = img1.cuda()
        dist01 = loss_fn.forward(img0,img1).item()
        
        mse_list.append(mse)
        ssim_list.append(ssim)
        psnr_list.append(psnr)
        lpips_list.append(dist01)

    print('Iteration2 Poisson :: MSE: {:.02f}, SSIM: {:.02f}, PSNR: {:.02f}, LPIPS: {:.02f}'.format(mean(mse_list), mean(ssim_list), mean(psnr_list), mean(lpips_list)))

    mse_list, ssim_list, psnr_list, lpips_list = [], [], [], []

    loss_fn = lpips.LPIPS(net='alex',version='0.1')
    loss_fn.cuda()

    for fR in tqdm(files):
        imgR = cv2.imread(fR, 1)
        fS = fR.replace('real', 'synthetic_cp')
        imgS = cv2.imread(fS, 1)
        assert imgR.shape == imgS.shape

        if imgR.shape[0] < 7 or imgR.shape[1] < 7:
            continue

        mse = mean_squared_error(imgR, imgS)
        if mse == 0.0:
            # cv2.imshow('img', np.hstack((imgR, imgS)))
            # key = cv2.waitKey(0)
            # if key == 27:
            #     cv2.destroyAllWindows()
            #     exit()
            continue
        ssim = structural_similarity(imgR, imgS, channel_axis=2)
        psnr = peak_signal_noise_ratio(imgR, imgS)

        imgR = cv2.resize(imgR, (64, 64), None, interpolation=cv2.INTER_AREA)
        imgS = cv2.resize(imgS, (64, 64), None, interpolation=cv2.INTER_AREA)

        img0 = lpips.im2tensor(imgR[:,:,::-1])
        img1 = lpips.im2tensor(imgS[:,:,::-1])
        img0 = img0.cuda()
        img1 = img1.cuda()
        dist01 = loss_fn.forward(img0,img1).item()
        
        mse_list.append(mse)
        ssim_list.append(ssim)
        psnr_list.append(psnr)
        lpips_list.append(dist01)

    print('Iteration2 Copy-Paste :: MSE: {:.02f}, SSIM: {:.02f}, PSNR: {:.02f}, LPIPS: {:.02f}'.format(mean(mse_list), mean(ssim_list), mean(psnr_list), mean(lpips_list)))


def main3():
    files = sorted(glob.glob('/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/results/DFG_resultcrops/iter3/real/*.png'))
    print('No. of files: ', len(files))

    mse_list, ssim_list, psnr_list, lpips_list = [], [], [], []

    loss_fn = lpips.LPIPS(net='alex',version='0.1')
    loss_fn.cuda()

    for fR in tqdm(files):
        imgR = cv2.imread(fR, 1)
        fS = fR.replace('real', 'synthetic_poisson')
        imgS = cv2.imread(fS, 1)
        assert imgR.shape == imgS.shape

        if imgR.shape[0] < 7 or imgR.shape[1] < 7:
            continue

        mse = mean_squared_error(imgR, imgS)
        if mse == 0.0:
            # cv2.imshow('img', np.hstack((imgR, imgS)))
            # key = cv2.waitKey(0)
            # if key == 27:
            #     cv2.destroyAllWindows()
            #     exit()
            continue
        ssim = structural_similarity(imgR, imgS, channel_axis=2)
        psnr = peak_signal_noise_ratio(imgR, imgS)

        imgR = cv2.resize(imgR, (64, 64), None, interpolation=cv2.INTER_AREA)
        imgS = cv2.resize(imgS, (64, 64), None, interpolation=cv2.INTER_AREA)

        img0 = lpips.im2tensor(imgR[:,:,::-1])
        img1 = lpips.im2tensor(imgS[:,:,::-1])
        img0 = img0.cuda()
        img1 = img1.cuda()
        dist01 = loss_fn.forward(img0,img1).item()
        
        mse_list.append(mse)
        ssim_list.append(ssim)
        psnr_list.append(psnr)
        lpips_list.append(dist01)

    print('Iteration3 Poisson :: MSE: {:.02f}, SSIM: {:.02f}, PSNR: {:.02f}, LPIPS: {:.02f}'.format(mean(mse_list), mean(ssim_list), mean(psnr_list), mean(lpips_list)))

    mse_list, ssim_list, psnr_list, lpips_list = [], [], [], []

    loss_fn = lpips.LPIPS(net='alex',version='0.1')
    loss_fn.cuda()

    for fR in tqdm(files):
        imgR = cv2.imread(fR, 1)
        fS = fR.replace('real', 'synthetic_cp')
        imgS = cv2.imread(fS, 1)
        assert imgR.shape == imgS.shape

        if imgR.shape[0] < 7 or imgR.shape[1] < 7:
            continue

        mse = mean_squared_error(imgR, imgS)
        if mse == 0.0:
            # cv2.imshow('img', np.hstack((imgR, imgS)))
            # key = cv2.waitKey(0)
            # if key == 27:
            #     cv2.destroyAllWindows()
            #     exit()
            continue
        ssim = structural_similarity(imgR, imgS, channel_axis=2)
        psnr = peak_signal_noise_ratio(imgR, imgS)

        imgR = cv2.resize(imgR, (64, 64), None, interpolation=cv2.INTER_AREA)
        imgS = cv2.resize(imgS, (64, 64), None, interpolation=cv2.INTER_AREA)

        img0 = lpips.im2tensor(imgR[:,:,::-1])
        img1 = lpips.im2tensor(imgS[:,:,::-1])
        img0 = img0.cuda()
        img1 = img1.cuda()
        dist01 = loss_fn.forward(img0,img1).item()
        
        mse_list.append(mse)
        ssim_list.append(ssim)
        psnr_list.append(psnr)
        lpips_list.append(dist01)

    print('Iteration3 Copy-Paste :: MSE: {:.02f}, SSIM: {:.02f}, PSNR: {:.02f}, LPIPS: {:.02f}'.format(mean(mse_list), mean(ssim_list), mean(psnr_list), mean(lpips_list)))

def main4():
    files = sorted(glob.glob('/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/results/DFG_resultcrops/iter4/real/*.png'))
    print('No. of files: ', len(files))

    mse_list, ssim_list, psnr_list, lpips_list = [], [], [], []

    loss_fn = lpips.LPIPS(net='alex',version='0.1')
    loss_fn.cuda()

    for fR in tqdm(files):
        imgR = cv2.imread(fR, 1)
        fS = fR.replace('real', 'synthetic_rainnet')
        imgS = cv2.imread(fS, 1)
        assert imgR.shape == imgS.shape

        if imgR.shape[0] < 7 or imgR.shape[1] < 7:
            continue

        mse = mean_squared_error(imgR, imgS)
        if mse == 0.0:
            # cv2.imshow('img', np.hstack((imgR, imgS)))
            # key = cv2.waitKey(0)
            # if key == 27:
            #     cv2.destroyAllWindows()
            #     exit()
            continue
        ssim = structural_similarity(imgR, imgS, channel_axis=2)
        psnr = peak_signal_noise_ratio(imgR, imgS)

        imgR = cv2.resize(imgR, (64, 64), None, interpolation=cv2.INTER_AREA)
        imgS = cv2.resize(imgS, (64, 64), None, interpolation=cv2.INTER_AREA)

        img0 = lpips.im2tensor(imgR[:,:,::-1])
        img1 = lpips.im2tensor(imgS[:,:,::-1])
        img0 = img0.cuda()
        img1 = img1.cuda()
        dist01 = loss_fn.forward(img0,img1).item()
        
        mse_list.append(mse)
        ssim_list.append(ssim)
        psnr_list.append(psnr)
        lpips_list.append(dist01)

    print('Iteration4 RainNet :: MSE: {:.02f}, SSIM: {:.02f}, PSNR: {:.02f}, LPIPS: {:.02f}'.format(mean(mse_list), mean(ssim_list), mean(psnr_list), mean(lpips_list)))

if __name__ == '__main__':
    main4()