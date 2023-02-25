import os, sys
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
import albumentations as A
from copy import deepcopy
import uuid
import random
import glob
from statistics import mean
import math
import argparse
import matplotlib.cm as cm

import torch

from src.utils import grid_sample
from src.easyaug import GeoAugParam, RandomGeoAug, RandomPhotoAug
import kornia
from typing import Tuple, Optional
from kornia.utils.helpers import _torch_inverse_cast

from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, make_matching_plot_fast, process_image)

torch.set_grad_enabled(False)

'''
Major parts borrowed from https://github.com/facebookresearch/theseus/blob/main/examples/homography_estimation.py
'''

parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--input0', type=str, default='/media/sayan/sayan/projects/traffic_sign/SuperGluePretrainedNetwork/assets/crops/I-8/5270.png',
    help='Path to the directory that contains the images')
parser.add_argument(
    '--input1', type=str, default='/media/sayan/sayan/projects/traffic_sign/SuperGluePretrainedNetwork/assets/crops/I-8/8013.png',
    help='Path to the directory that contains the images')
parser.add_argument(
    '--resize', type=int, nargs='+', default=[-1],
    help='Resize the input image before running inference. If two numbers, '
            'resize to the exact dimensions, if one number, resize the max '
            'dimension, if -1, do not resize')
parser.add_argument(
    '--resize_float', action='store_true',
    help='Resize the image after casting uint8 to float')

parser.add_argument(
    '--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
    help='SuperGlue weights')
parser.add_argument(
    '--max_keypoints', type=int, default=-1,
    help='Maximum number of keypoints detected by Superpoint'
            ' (\'-1\' keeps all keypoints)')
parser.add_argument(
    '--keypoint_threshold', type=float, default=0.005,
    help='SuperPoint keypoint detector confidence threshold')
parser.add_argument(
    '--nms_radius', type=int, default=2,
    help='SuperPoint Non Maximum Suppression (NMS) radius'
    ' (Must be positive)')
parser.add_argument(
    '--sinkhorn_iterations', type=int, default=20,
    help='Number of Sinkhorn iterations performed by SuperGlue')
parser.add_argument(
    '--match_threshold', type=float, default=0.1,
    help='SuperGlue match threshold')

parser.add_argument(
    '--force_cpu', action='store_true',
    help='Force pytorch to run in CPU mode.')
opt = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
print('Running inference on device \"{}\"'.format(device))
config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
matching = Matching(config).eval().to(device)

def warp_perspective_norm(H, img):
    height, width = img.shape[-2:]
    grid = kornia.utils.create_meshgrid(
        height, width, normalized_coordinates=True, device=H.device
    )
    Hinv = torch.inverse(H)
    warped_grid = kornia.geometry.transform.homography_warper.warp_grid(grid, Hinv)
    # Using custom implementation, above will throw error with outer loop optim.
    img2 = grid_sample(img, warped_grid)
    return img2

def normalize_homography(
    dst_pix_trans_src_pix: torch.Tensor, dsize_src: Tuple[int, int], dsize_dst: Tuple[int, int]
) -> torch.Tensor:
    r"""Normalize a given homography in pixels to [-1, 1].
    Args:
        dst_pix_trans_src_pix (torch.Tensor): homography/ies from source to destination to be
          normalized. :math:`(B, 3, 3)`
        dsize_src (tuple): size of the source image (height, width).
        dsize_dst (tuple): size of the destination image (height, width).
    Returns:
        torch.Tensor: the normalized homography of shape :math:`(B, 3, 3)`.
    """
    # check_is_tensor(dst_pix_trans_src_pix)

    if not (len(dst_pix_trans_src_pix.shape) == 3 or dst_pix_trans_src_pix.shape[-2:] == (3, 3)):
        raise ValueError(
            "Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {}".format(dst_pix_trans_src_pix.shape)
        )

    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel(src_h, src_w).to(dst_pix_trans_src_pix)

    src_pix_trans_src_norm = _torch_inverse_cast(src_norm_trans_src_pix)
    dst_norm_trans_dst_pix: torch.Tensor = normal_transform_pixel(dst_h, dst_w).to(dst_pix_trans_src_pix)

    # compute chain transformations
    dst_norm_trans_src_norm: torch.Tensor = dst_norm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm)
    return dst_norm_trans_src_norm

def normal_transform_pixel(
    height: int,
    width: int,
    eps: float = 1e-14,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Compute the normalization matrix from image size in pixels to [-1, 1].
    Args:
        height (int): image height.
        width (int): image width.
        eps (float): epsilon to prevent divide-by-zero errors
    Returns:
        torch.Tensor: normalized transform with shape :math:`(1, 3, 3)`.
    """
    tr_mat = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)  # 3x3

    # prevent divide by zero bugs
    width_denom: float = eps if width == 1 else width - 1.0
    height_denom: float = eps if height == 1 else height - 1.0

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom

    return tr_mat.unsqueeze(0)  # 1x3x3

def denormalize_homography(norm_homo, height, width):
    normal_transform_pix = normal_transform_pixel(height, width, device=norm_homo.device)
    denormed_homo = torch.inverse(normal_transform_pix) @ norm_homo @ normal_transform_pix
    return denormed_homo

# L2 distance between 4 corners of source image warped using GT homography
# and estimated homography transform '
# (Mean Average Corner Error Deep Image Homography Estimation https://arxiv.org/pdf/1606.03798.pdf)
# 
def four_corner_dist(H_1_2, H_1_2_gt, height, width):
    Hinv_gt = torch.inverse(H_1_2_gt)
    Hinv = torch.inverse(H_1_2)
    grid = kornia.utils.create_meshgrid(2, 2, device=Hinv.device)
    warped_grid = kornia.geometry.transform.homography_warper.warp_grid(grid, Hinv)
    warped_grid_gt = kornia.geometry.transform.homography_warper.warp_grid(
        grid, Hinv_gt
    )
    warped_grid = (warped_grid + 1) / 2
    warped_grid_gt = (warped_grid_gt + 1) / 2
    warped_grid[..., 0] *= width
    warped_grid[..., 1] *= height
    warped_grid_gt[..., 0] *= width
    warped_grid_gt[..., 1] *= height
    dist = torch.norm(warped_grid - warped_grid_gt, p=2, dim=-1)
    dist = dist.mean(dim=-1).mean(dim=-1)
    return dist

def create_dataset():
    img_files = sorted(glob.glob('/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/crops/*/*.png'))

    sc = 0.3
    rga = RandomGeoAug(
        rotate_param=GeoAugParam(min=-30 * sc, max=30 * sc),
        scale_param=GeoAugParam(min=(1.0 - 0.8 * sc), max=(1.0 + 1.2 * sc)),
        translate_x_param=GeoAugParam(min=-0.2 * sc, max=0.2 * sc),
        translate_y_param=GeoAugParam(min=-0.2 * sc, max=0.2 * sc),
        shear_x_param=GeoAugParam(min=-10 * sc, max=10 * sc),
        shear_y_param=GeoAugParam(min=-10 * sc, max=10 * sc),
        perspective_param=GeoAugParam(min=-0.1 * sc, max=0.1 * sc),
    )

    rpa = RandomPhotoAug()
    prob = 0.2
    mag = 0.2
    rpa.set_all_probs(prob)
    rpa.set_all_mags(mag)

    for f in tqdm(img_files):
        img1_bgra = cv2.imread(f, -1)
        img1_bgra = cv2.copyMakeBorder(img1_bgra,64,64,64,64,cv2.BORDER_CONSTANT,value=(0, 0, 0))

        for i in range(3):
            img1 = cv2.cvtColor(img1_bgra, cv2.COLOR_BGRA2BGR)
            
            # random geometric augmentation
            img1_torch = torch.from_numpy(img1.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
            _, H_1_2_norm = rga.forward(
                img1_torch, return_transform=True, normalize_returned_transform=True
            )

            H_1_2 = denormalize_homography(H_1_2_norm, img1.shape[0], img1.shape[1])
            # H_1_2_norm = normalize_homography(H_1_2, [img1.shape[0], img1.shape[1]], [img1.shape[0], img1.shape[1]])
            H_1_2 = H_1_2.squeeze(0).numpy()

            img2_bgra = cv2.warpPerspective(img1_bgra, H_1_2, (img1.shape[1],img1.shape[0]))
            img2 = cv2.cvtColor(img2_bgra, cv2.COLOR_BGRA2BGR)
            img2_torch = torch.from_numpy(img2.astype(np.float32) / 255.0).permute(2, 0, 1)[None]

            # random photometric augmentation
            img1_torch = torch.clamp(img1_torch, 0.0, 1.0)
            img2_torch = torch.clamp(img2_torch, 0.0, 1.0)
            img1_torch = rpa.forward(img1_torch)
            img2_torch = rpa.forward(img2_torch)

            img1_torch = img1_torch.squeeze(0).permute(1, 2, 0)
            img1 = img1_torch.numpy()
            img1 = img1 * 255.0
            img1 = img1.astype(np.uint8)

            img2_torch = img2_torch.squeeze(0).permute(1, 2, 0)
            img2 = img2_torch.numpy()
            img2 = img2 * 255.0
            img2 = img2.astype(np.uint8)

            b, g, r = cv2.split(img1)
            _, _, _, a = cv2.split(img1_bgra)
            a_t = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
            temp1 = cv2.merge((b, g, r))
            bg = np.zeros_like(temp1, np.uint8) * 255
            fg = temp1.astype(float) / 255.0
            bg = bg.astype(float) / 255.0
            a_t = a_t.astype(float) / 255.0
            fg = cv2.multiply(a_t, fg)
            bg = cv2.multiply(1.0 - a_t, bg)
            temp2 = cv2.add(fg, bg)
            temp2 = temp2 * 255.0
            temp2 = temp2.astype(np.uint8)
            b, g, r = cv2.split(temp2)
            img1_bgra = cv2.merge((b, g, r, a))

            b, g, r = cv2.split(img2)
            _, _, _, a = cv2.split(img2_bgra)
            a_t = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
            temp1 = cv2.merge((b, g, r))
            bg = np.zeros_like(temp1, np.uint8) * 255
            fg = temp1.astype(float) / 255.0
            bg = bg.astype(float) / 255.0
            a_t = a_t.astype(float) / 255.0
            fg = cv2.multiply(a_t, fg)
            bg = cv2.multiply(1.0 - a_t, bg)
            temp2 = cv2.add(fg, bg)
            temp2 = temp2 * 255.0
            temp2 = temp2.astype(np.uint8)
            b, g, r = cv2.split(temp2)
            img2_bgra = cv2.merge((b, g, r, a))

            # cv2.imshow('asd', np.hstack((img1_bgra, img2_bgra)))
            # key = cv2.waitKey(0)
            # if key == 27:
            #     cv2.destroyAllWindows()
            #     exit()

            fname = '/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/benchmark_homography/img1/{}_{}_{:03d}.png'.format(f.split('/')[-2], f.split('/')[-1][:-4], i)
            cv2.imwrite(fname, img1_bgra)
            fname = fname.replace('img1', 'img2')
            cv2.imwrite(fname, img2_bgra)
            fname = fname.replace('img2', 'H_1_2').replace('.png', '.npy')
            np.save(fname, H_1_2)

def bm_sift(black=True):
    img_files = sorted(glob.glob('/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/benchmark_homography/img1/*.png'))

    mace = []
    misses = 0
    match_count = []

    for f_img1 in tqdm(img_files):
        img1_bgra = cv2.imread(f_img1, -1)
        f_img2 = f_img1.replace('img1', 'img2')
        img2_bgra = cv2.imread(f_img2, -1)
        f_h = f_img1.replace('img1', 'H_1_2').replace('.png', '.npy')
        H_1_2_gt = np.load(f_h)

        if not black:
            b, g, r, a = cv2.split(img1_bgra)
            a_t = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
            temp1 = cv2.merge((b, g, r))
            bg = np.ones_like(temp1, np.uint8) * 255
            fg = temp1.astype(float) / 255.0
            bg = bg.astype(float) / 255.0
            a_t = a_t.astype(float) / 255.0
            fg = cv2.multiply(a_t, fg)
            bg = cv2.multiply(1.0 - a_t, bg)
            temp2 = cv2.add(fg, bg)
            temp2 = temp2 * 255.0
            temp2 = temp2.astype(np.uint8)
            b, g, r = cv2.split(temp2)
            img1_bgra = cv2.merge((b, g, r, a))

            b, g, r, a = cv2.split(img2_bgra)
            a_t = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
            temp1 = cv2.merge((b, g, r))
            bg = np.ones_like(temp1, np.uint8) * 255
            fg = temp1.astype(float) / 255.0
            bg = bg.astype(float) / 255.0
            a_t = a_t.astype(float) / 255.0
            fg = cv2.multiply(a_t, fg)
            bg = cv2.multiply(1.0 - a_t, bg)
            temp2 = cv2.add(fg, bg)
            temp2 = temp2 * 255.0
            temp2 = temp2.astype(np.uint8)
            b, g, r = cv2.split(temp2)
            img2_bgra = cv2.merge((b, g, r, a))

        # Initiate SIFT detector
        sift = cv2.SIFT_create()

        img1_gray = cv2.cvtColor(img1_bgra, cv2.COLOR_BGRA2GRAY)
        img2_gray = cv2.cvtColor(img2_bgra, cv2.COLOR_BGRA2GRAY)

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1_gray,None)
        kp2, des2 = sift.detectAndCompute(img2_gray,None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        try:
            matches = flann.knnMatch(des1,des2,k=2)
        except:
            # print('des1: {}, des2: {}'.format(des1, des2))
            misses += 1
            continue
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        if len(good) >= 4:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            H_1_2_pred, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
            img3_bgra = cv2.drawMatches(img1_bgra,kp1,img2_bgra,kp2,good,None,**draw_params)

            h,w = img1_gray.shape
            pts = np.float32([ [w//2-30,h//2-30],[w//2+30,h//2-30],[w//2+30,h//2+30],[w//2-30,h//2+30] ]).reshape(-1,1,2)
            dst_gt = cv2.perspectiveTransform(pts,H_1_2_gt)
            try:
                dst_pred = cv2.perspectiveTransform(pts,H_1_2_pred)
            except:
                # print('H_1_2_pred: ', H_1_2_pred)
                misses += 1
                continue
            img2_bgra_pred = cv2.warpPerspective(img1_bgra,H_1_2_pred, (img2_bgra.shape[1],img2_bgra.shape[0]))
            img2_bgra_pred = cv2.polylines(img2_bgra_pred,[np.int32(dst_gt)],True,(0, 255, 0),2, cv2.LINE_AA)
            img2_bgra_pred = cv2.polylines(img2_bgra_pred,[np.int32(dst_pred)],True,(0, 0, 255),2, cv2.LINE_AA)

            H_1_2_gt = torch.from_numpy(H_1_2_gt).unsqueeze(0)
            H_1_2_pred = torch.from_numpy(H_1_2_pred).unsqueeze(0)
            H_1_2_gt_norm = normalize_homography(H_1_2_gt, [img1_bgra.shape[0], img1_bgra.shape[1]], [img1_bgra.shape[0], img1_bgra.shape[1]])
            H_1_2_pred_norm = normalize_homography(H_1_2_pred, [img1_bgra.shape[0], img1_bgra.shape[1]], [img1_bgra.shape[0], img1_bgra.shape[1]])

            corner_err = four_corner_dist(H_1_2_pred_norm, H_1_2_gt_norm, img1_bgra.shape[0], img1_bgra.shape[1]).item()
            mace.append(corner_err)
            match_count.append(len(good))
            if math.isnan(corner_err):
                misses += 1
                # cv2.imshow('asd', np.hstack((img3_bgra, img2_bgra_pred)))
                
                print('corner_err: {}, matches: {}'.format(corner_err, len(good)))

                # key = cv2.waitKey(0)
                # if key == 27:
                #     cv2.destroyAllWindows()
                #     exit()
        else:
            misses += 1
        
    print('SIFT+FLANN MACE: {}, Miss: {}, Average matches: {}'.format(np.nanmean(np.array(mace)), misses, mean(match_count)))

def bm_orb(black=True):
    img_files = sorted(glob.glob('/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/benchmark_homography/img1/*.png'))

    mace = []
    misses = 0
    match_count = []

    for f_img1 in tqdm(img_files):
        img1_bgra = cv2.imread(f_img1, -1)
        f_img2 = f_img1.replace('img1', 'img2')
        img2_bgra = cv2.imread(f_img2, -1)
        f_h = f_img1.replace('img1', 'H_1_2').replace('.png', '.npy')
        H_1_2_gt = np.load(f_h)

        if not black:
            b, g, r, a = cv2.split(img1_bgra)
            a_t = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
            temp1 = cv2.merge((b, g, r))
            bg = np.ones_like(temp1, np.uint8) * 255
            fg = temp1.astype(float) / 255.0
            bg = bg.astype(float) / 255.0
            a_t = a_t.astype(float) / 255.0
            fg = cv2.multiply(a_t, fg)
            bg = cv2.multiply(1.0 - a_t, bg)
            temp2 = cv2.add(fg, bg)
            temp2 = temp2 * 255.0
            temp2 = temp2.astype(np.uint8)
            b, g, r = cv2.split(temp2)
            img1_bgra = cv2.merge((b, g, r, a))

            b, g, r, a = cv2.split(img2_bgra)
            a_t = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
            temp1 = cv2.merge((b, g, r))
            bg = np.ones_like(temp1, np.uint8) * 255
            fg = temp1.astype(float) / 255.0
            bg = bg.astype(float) / 255.0
            a_t = a_t.astype(float) / 255.0
            fg = cv2.multiply(a_t, fg)
            bg = cv2.multiply(1.0 - a_t, bg)
            temp2 = cv2.add(fg, bg)
            temp2 = temp2 * 255.0
            temp2 = temp2.astype(np.uint8)
            b, g, r = cv2.split(temp2)
            img2_bgra = cv2.merge((b, g, r, a))

        # Initiate detector
        orb = cv2.ORB_create(nfeatures=2000)

        img1_gray = cv2.cvtColor(img1_bgra, cv2.COLOR_BGRA2GRAY)
        img2_gray = cv2.cvtColor(img2_bgra, cv2.COLOR_BGRA2GRAY)

        # find the keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(img1_gray,None)
        kp2, des2 = orb.detectAndCompute(img2_gray,None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        try:
            matches = bf.match(des1, des2)
            dmatches = sorted(matches, key = lambda x:x.distance)
        except:
            # print('des1: {}, des2: {}'.format(des1, des2))
            misses += 1
            continue
        if len(dmatches)>=4:
            src_pts  = np.float32([kp1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
            dst_pts  = np.float32([kp2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)
            
            H_1_2_pred, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
            img3_bgra = cv2.drawMatches(img1_bgra,kp1,img2_bgra,kp2,dmatches,None,**draw_params)

            h,w = img1_gray.shape
            pts = np.float32([ [w//2-30,h//2-30],[w//2+30,h//2-30],[w//2+30,h//2+30],[w//2-30,h//2+30] ]).reshape(-1,1,2)
            dst_gt = cv2.perspectiveTransform(pts,H_1_2_gt)
            try:
                dst_pred = cv2.perspectiveTransform(pts,H_1_2_pred)
            except:
                # print('H_1_2_pred: ', H_1_2_pred)
                misses += 1
                continue
            img2_bgra_pred = cv2.warpPerspective(img1_bgra,H_1_2_pred, (img2_bgra.shape[1],img2_bgra.shape[0]))
            img2_bgra_pred = cv2.polylines(img2_bgra_pred,[np.int32(dst_gt)],True,(0, 255, 0),2, cv2.LINE_AA)
            img2_bgra_pred = cv2.polylines(img2_bgra_pred,[np.int32(dst_pred)],True,(0, 0, 255),2, cv2.LINE_AA)

            H_1_2_gt = torch.from_numpy(H_1_2_gt).unsqueeze(0)
            H_1_2_pred = torch.from_numpy(H_1_2_pred).unsqueeze(0)
            H_1_2_gt_norm = normalize_homography(H_1_2_gt, [img1_bgra.shape[0], img1_bgra.shape[1]], [img1_bgra.shape[0], img1_bgra.shape[1]])
            H_1_2_pred_norm = normalize_homography(H_1_2_pred, [img1_bgra.shape[0], img1_bgra.shape[1]], [img1_bgra.shape[0], img1_bgra.shape[1]])

            corner_err = four_corner_dist(H_1_2_pred_norm, H_1_2_gt_norm, img1_bgra.shape[0], img1_bgra.shape[1]).item()
            mace.append(corner_err)
            match_count.append(len(dmatches))
            if math.isnan(corner_err):
                misses += 1
                print('corner_err: {}, matches: {}'.format(corner_err, len(dmatches)))

            # cv2.imshow('asd', np.hstack((img3_bgra, img2_bgra_pred)))
            # key = cv2.waitKey(0)
            # if key == 27:
            #     cv2.destroyAllWindows()
            #     exit()
            # elif key == ord('s'):
            #     spath = './results/orb/{}'.format(f_img1.split('/')[-1])
            #     cv2.imwrite(spath, np.hstack((img3_bgra, img2_bgra_pred)))
        else:
            misses += 1
        
    print('ORB+BFMatcher MACE: {}, Miss: {}, Average matches: {}'.format(np.nanmean(np.array(mace)), misses, mean(match_count)))

def bm_akaze(black=True):
    img_files = sorted(glob.glob('/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/benchmark_homography/img1/*.png'))

    mace = []
    misses = 0
    match_count = []

    for f_img1 in tqdm(img_files):
        img1_bgra = cv2.imread(f_img1, -1)
        f_img2 = f_img1.replace('img1', 'img2')
        img2_bgra = cv2.imread(f_img2, -1)
        f_h = f_img1.replace('img1', 'H_1_2').replace('.png', '.npy')
        H_1_2_gt = np.load(f_h)

        if not black:
            b, g, r, a = cv2.split(img1_bgra)
            a_t = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
            temp1 = cv2.merge((b, g, r))
            bg = np.ones_like(temp1, np.uint8) * 255
            fg = temp1.astype(float) / 255.0
            bg = bg.astype(float) / 255.0
            a_t = a_t.astype(float) / 255.0
            fg = cv2.multiply(a_t, fg)
            bg = cv2.multiply(1.0 - a_t, bg)
            temp2 = cv2.add(fg, bg)
            temp2 = temp2 * 255.0
            temp2 = temp2.astype(np.uint8)
            b, g, r = cv2.split(temp2)
            img1_bgra = cv2.merge((b, g, r, a))

            b, g, r, a = cv2.split(img2_bgra)
            a_t = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
            temp1 = cv2.merge((b, g, r))
            bg = np.ones_like(temp1, np.uint8) * 255
            fg = temp1.astype(float) / 255.0
            bg = bg.astype(float) / 255.0
            a_t = a_t.astype(float) / 255.0
            fg = cv2.multiply(a_t, fg)
            bg = cv2.multiply(1.0 - a_t, bg)
            temp2 = cv2.add(fg, bg)
            temp2 = temp2 * 255.0
            temp2 = temp2.astype(np.uint8)
            b, g, r = cv2.split(temp2)
            img2_bgra = cv2.merge((b, g, r, a))

        # Initiate detector
        akaze = cv2.AKAZE_create()

        img1_gray = cv2.cvtColor(img1_bgra, cv2.COLOR_BGRA2GRAY)
        img2_gray = cv2.cvtColor(img2_bgra, cv2.COLOR_BGRA2GRAY)

        # find the keypoints and descriptors
        kp1, des1 = akaze.detectAndCompute(img1_gray,None)
        kp2, des2 = akaze.detectAndCompute(img2_gray,None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        try:
            matches = bf.match(des1, des2)
            dmatches = sorted(matches, key = lambda x:x.distance)
        except:
            # print('des1: {}, des2: {}'.format(des1, des2))
            misses += 1
            continue
        if len(dmatches)>=4:
            src_pts  = np.float32([kp1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
            dst_pts  = np.float32([kp2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)
            
            H_1_2_pred, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
            img3_bgra = cv2.drawMatches(img1_bgra,kp1,img2_bgra,kp2,dmatches,None,**draw_params)

            h,w = img1_gray.shape
            pts = np.float32([ [w//2-30,h//2-30],[w//2+30,h//2-30],[w//2+30,h//2+30],[w//2-30,h//2+30] ]).reshape(-1,1,2)
            dst_gt = cv2.perspectiveTransform(pts,H_1_2_gt)
            try:
                dst_pred = cv2.perspectiveTransform(pts,H_1_2_pred)
            except:
                # print('H_1_2_pred: ', H_1_2_pred)
                misses += 1
                continue
            img2_bgra_pred = cv2.warpPerspective(img1_bgra,H_1_2_pred, (img2_bgra.shape[1],img2_bgra.shape[0]))
            img2_bgra_pred = cv2.polylines(img2_bgra_pred,[np.int32(dst_gt)],True,(0, 255, 0),2, cv2.LINE_AA)
            img2_bgra_pred = cv2.polylines(img2_bgra_pred,[np.int32(dst_pred)],True,(0, 0, 255),2, cv2.LINE_AA)

            H_1_2_gt = torch.from_numpy(H_1_2_gt).unsqueeze(0)
            H_1_2_pred = torch.from_numpy(H_1_2_pred).unsqueeze(0)
            H_1_2_gt_norm = normalize_homography(H_1_2_gt, [img1_bgra.shape[0], img1_bgra.shape[1]], [img1_bgra.shape[0], img1_bgra.shape[1]])
            H_1_2_pred_norm = normalize_homography(H_1_2_pred, [img1_bgra.shape[0], img1_bgra.shape[1]], [img1_bgra.shape[0], img1_bgra.shape[1]])

            corner_err = four_corner_dist(H_1_2_pred_norm, H_1_2_gt_norm, img1_bgra.shape[0], img1_bgra.shape[1]).item()
            mace.append(corner_err)
            match_count.append(len(dmatches))
            if math.isnan(corner_err):
                misses += 1
                print('corner_err: {}, matches: {}'.format(corner_err, len(dmatches)))

            # cv2.imshow('asd', np.hstack((img3_bgra, img2_bgra_pred)))
            # key = cv2.waitKey(0)
            # if key == 27:
            #     cv2.destroyAllWindows()
            #     exit()
        else:
            misses += 1
        
    print('AKAZE+BFMatcher MACE: {}, Miss: {}, Average matches: {}'.format(np.nanmean(np.array(mace)), misses, mean(match_count)))

def bm_brisk(black=True):
    img_files = sorted(glob.glob('/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/benchmark_homography/img1/*.png'))

    mace = []
    misses = 0
    match_count = []

    for f_img1 in tqdm(img_files):
        img1_bgra = cv2.imread(f_img1, -1)
        f_img2 = f_img1.replace('img1', 'img2')
        img2_bgra = cv2.imread(f_img2, -1)
        f_h = f_img1.replace('img1', 'H_1_2').replace('.png', '.npy')
        H_1_2_gt = np.load(f_h)

        if not black:
            b, g, r, a = cv2.split(img1_bgra)
            a_t = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
            temp1 = cv2.merge((b, g, r))
            bg = np.ones_like(temp1, np.uint8) * 255
            fg = temp1.astype(float) / 255.0
            bg = bg.astype(float) / 255.0
            a_t = a_t.astype(float) / 255.0
            fg = cv2.multiply(a_t, fg)
            bg = cv2.multiply(1.0 - a_t, bg)
            temp2 = cv2.add(fg, bg)
            temp2 = temp2 * 255.0
            temp2 = temp2.astype(np.uint8)
            b, g, r = cv2.split(temp2)
            img1_bgra = cv2.merge((b, g, r, a))

            b, g, r, a = cv2.split(img2_bgra)
            a_t = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
            temp1 = cv2.merge((b, g, r))
            bg = np.ones_like(temp1, np.uint8) * 255
            fg = temp1.astype(float) / 255.0
            bg = bg.astype(float) / 255.0
            a_t = a_t.astype(float) / 255.0
            fg = cv2.multiply(a_t, fg)
            bg = cv2.multiply(1.0 - a_t, bg)
            temp2 = cv2.add(fg, bg)
            temp2 = temp2 * 255.0
            temp2 = temp2.astype(np.uint8)
            b, g, r = cv2.split(temp2)
            img2_bgra = cv2.merge((b, g, r, a))

        # Initiate detector
        brisk = cv2.BRISK_create()

        img1_gray = cv2.cvtColor(img1_bgra, cv2.COLOR_BGRA2GRAY)
        img2_gray = cv2.cvtColor(img2_bgra, cv2.COLOR_BGRA2GRAY)

        # find the keypoints and descriptors
        kp1, des1 = brisk.detectAndCompute(img1_gray,None)
        kp2, des2 = brisk.detectAndCompute(img2_gray,None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        try:
            matches = bf.match(des1, des2)
            dmatches = sorted(matches, key = lambda x:x.distance)
        except:
            # print('des1: {}, des2: {}'.format(des1, des2))
            misses += 1
            continue
        if len(dmatches)>=4:
            src_pts  = np.float32([kp1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
            dst_pts  = np.float32([kp2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)
            
            H_1_2_pred, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
            img3_bgra = cv2.drawMatches(img1_bgra,kp1,img2_bgra,kp2,dmatches,None,**draw_params)

            h,w = img1_gray.shape
            pts = np.float32([ [w//2-30,h//2-30],[w//2+30,h//2-30],[w//2+30,h//2+30],[w//2-30,h//2+30] ]).reshape(-1,1,2)
            dst_gt = cv2.perspectiveTransform(pts,H_1_2_gt)
            try:
                dst_pred = cv2.perspectiveTransform(pts,H_1_2_pred)
            except:
                # print('H_1_2_pred: ', H_1_2_pred)
                misses += 1
                continue
            img2_bgra_pred = cv2.warpPerspective(img1_bgra,H_1_2_pred, (img2_bgra.shape[1],img2_bgra.shape[0]))
            img2_bgra_pred = cv2.polylines(img2_bgra_pred,[np.int32(dst_gt)],True,(0, 255, 0),2, cv2.LINE_AA)
            img2_bgra_pred = cv2.polylines(img2_bgra_pred,[np.int32(dst_pred)],True,(0, 0, 255),2, cv2.LINE_AA)

            H_1_2_gt = torch.from_numpy(H_1_2_gt).unsqueeze(0)
            H_1_2_pred = torch.from_numpy(H_1_2_pred).unsqueeze(0)
            H_1_2_gt_norm = normalize_homography(H_1_2_gt, [img1_bgra.shape[0], img1_bgra.shape[1]], [img1_bgra.shape[0], img1_bgra.shape[1]])
            H_1_2_pred_norm = normalize_homography(H_1_2_pred, [img1_bgra.shape[0], img1_bgra.shape[1]], [img1_bgra.shape[0], img1_bgra.shape[1]])

            corner_err = four_corner_dist(H_1_2_pred_norm, H_1_2_gt_norm, img1_bgra.shape[0], img1_bgra.shape[1]).item()
            mace.append(corner_err)
            match_count.append(len(dmatches))
            if math.isnan(corner_err):
                misses += 1
                print('corner_err: {}, matches: {}'.format(corner_err, len(dmatches)))

            # cv2.imshow('asd', np.hstack((img3_bgra, img2_bgra_pred)))
            # key = cv2.waitKey(0)
            # if key == 27:
            #     cv2.destroyAllWindows()
            #     exit()
        else:
            misses += 1
        
    print('BRISK+BFMatcher MACE: {}, Miss: {}, Average matches: {}'.format(np.nanmean(np.array(mace)), misses, mean(match_count)))

def bm_superglue(black=True):
    img_files = sorted(glob.glob('/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/benchmark_homography/img1/*.png'))

    mace = []
    misses = 0
    match_count = []

    for f_img1 in tqdm(img_files):
        img1_bgra = cv2.imread(f_img1, -1)
        f_img2 = f_img1.replace('img1', 'img2')
        img2_bgra = cv2.imread(f_img2, -1)
        f_h = f_img1.replace('img1', 'H_1_2').replace('.png', '.npy')
        H_1_2_gt = np.load(f_h)

        if not black:
            b, g, r, a = cv2.split(img1_bgra)
            a_t = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
            temp1 = cv2.merge((b, g, r))
            bg = np.ones_like(temp1, np.uint8) * 255
            fg = temp1.astype(float) / 255.0
            bg = bg.astype(float) / 255.0
            a_t = a_t.astype(float) / 255.0
            fg = cv2.multiply(a_t, fg)
            bg = cv2.multiply(1.0 - a_t, bg)
            temp2 = cv2.add(fg, bg)
            temp2 = temp2 * 255.0
            temp2 = temp2.astype(np.uint8)
            b, g, r = cv2.split(temp2)
            img1_bgra = cv2.merge((b, g, r, a))

            b, g, r, a = cv2.split(img2_bgra)
            a_t = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
            temp1 = cv2.merge((b, g, r))
            bg = np.ones_like(temp1, np.uint8) * 255
            fg = temp1.astype(float) / 255.0
            bg = bg.astype(float) / 255.0
            a_t = a_t.astype(float) / 255.0
            fg = cv2.multiply(a_t, fg)
            bg = cv2.multiply(1.0 - a_t, bg)
            temp2 = cv2.add(fg, bg)
            temp2 = temp2 * 255.0
            temp2 = temp2.astype(np.uint8)
            b, g, r = cv2.split(temp2)
            img2_bgra = cv2.merge((b, g, r, a))

        image1_bgra, image1, inp1, scales1, h1, w1 = process_image(
        img1_bgra, device, opt.resize, 0, opt.resize_float)
        image2_bgra, image2, inp2, scales2, h2, w2 = process_image(
            img2_bgra, device, opt.resize, 0, opt.resize_float)
        
        pred = matching({'image0': inp1, 'image1': inp2})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        if len(mkpts0)>=4:
            mkpts0 = mkpts0.astype(np.int32)
            mkpts1 = mkpts1.astype(np.int32)

            H_1_2_pred, mask = cv2.findHomography(mkpts0, mkpts1)
            matchesMask = mask.ravel().tolist()

            color = cm.jet(conf[valid])
            text = [
                    'SuperGlue',
                    'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                    'Matches: {}'.format(len(mkpts0))
                ]
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            stem0 = int(float(opt.input0.split('/')[-1][:-4]))
            stem1 = int(float(opt.input1.split('/')[-1][:-4]))
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {:06}:{:06}'.format(stem0, stem1),
            ]
            img3_bgr = make_matching_plot_fast(
                image1, image2, kpts0, kpts1, mkpts0, mkpts1, color, text,
                path=None, show_keypoints=False, small_text=small_text, opencv_display=False)

            h,w,_ = img1_bgra.shape
            pts = np.float32([ [w//2-30,h//2-30],[w//2+30,h//2-30],[w//2+30,h//2+30],[w//2-30,h//2+30] ]).reshape(-1,1,2)
            dst_gt = cv2.perspectiveTransform(pts,H_1_2_gt)
            try:
                dst_pred = cv2.perspectiveTransform(pts,H_1_2_pred)
            except:
                # print('H_1_2_pred: ', H_1_2_pred)
                misses += 1
                continue
            img2_bgra_pred = cv2.warpPerspective(img1_bgra,H_1_2_pred, (img2_bgra.shape[1],img2_bgra.shape[0]))
            img2_bgra_pred = cv2.polylines(img2_bgra_pred,[np.int32(dst_gt)],True,(0, 255, 0),2, cv2.LINE_AA)
            img2_bgra_pred = cv2.polylines(img2_bgra_pred,[np.int32(dst_pred)],True,(0, 0, 255),2, cv2.LINE_AA)

            H_1_2_gt = torch.from_numpy(H_1_2_gt).unsqueeze(0)
            H_1_2_pred = torch.from_numpy(H_1_2_pred).unsqueeze(0)
            H_1_2_gt_norm = normalize_homography(H_1_2_gt, [img1_bgra.shape[0], img1_bgra.shape[1]], [img2_bgra.shape[0], img2_bgra.shape[1]])
            H_1_2_pred_norm = normalize_homography(H_1_2_pred, [img1_bgra.shape[0], img1_bgra.shape[1]], [img2_bgra.shape[0], img2_bgra.shape[1]])

            corner_err = four_corner_dist(H_1_2_pred_norm, H_1_2_gt_norm, img1_bgra.shape[0], img1_bgra.shape[1]).item()
            mace.append(corner_err)
            match_count.append(len(mkpts0))
            if math.isnan(corner_err):
                misses += 1
                print('corner_err: {}, matches: {}'.format(corner_err, len(mkpts0)))

            # cv2.imshow('res', img3_bgr)
            # cv2.imshow('asd', np.hstack((img1_bgra, img2_bgra, img2_bgra_pred)))
            # key = cv2.waitKey(0)
            # if key == 27:
            #     cv2.destroyAllWindows()
            #     exit()
        else:
            misses += 1
        
    print('SuperPoint+SuperGlue MACE: {}, Miss: {}, Average matches: {}'.format(np.nanmean(np.array(mace)), misses, mean(match_count)))

def main():
    # 1. Create image pairs and homography matrix
    # create_dataset()
    
    # black background
    # 2. Benchmark SIFT
    # bm_sift()
    # 3. Benchmark ORB
    # bm_orb()
    # 4. Benchmark AKAZE
    # bm_akaze()
    # 5. Benchmark BRISK
    # bm_brisk()
    # 6. Benchmark Superpoint+SuperGlue
    # bm_superglue()
    
    # white background
    # 7. Benchmark SIFT
    bm_sift(black=False)
    # 8. Benchmark ORB
    bm_orb(black=False)
    # 9. Benchmark AKAZE
    bm_akaze(black=False)
    # 10. Benchmark BRISK
    bm_brisk(black=False)
    # 11. Benchmark Superpoint+SuperGlue
    bm_superglue(black=False)

if __name__ == '__main__':
    main()