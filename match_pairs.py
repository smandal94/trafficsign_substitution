#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# Project: SuperGlue
# Paper: https://arxiv.org/pdf/1911.11763.pdf
# Code: https://github.com/magicleap/SuperGluePretrainedNetwork
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import cv2
from scipy.spatial import distance


from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, make_matching_plot_fast, process_image)

torch.set_grad_enabled(False)

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
    '--resize', type=int, nargs='+', default=[-1], # default=[256, 256],
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
    '--max_keypoints', type=int, default=-1, # default=1024,
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
'''
    From PyImageSearch :)
    https://github.com/PyImageSearch/imutils/blob/c12f15391fcc945d0d644b85194b8c044a392e0a/imutils/perspective.py
'''
def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = distance.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype=int) 


def align_superglue(template, img, new, category='None', pad=64):
    img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)
    template = cv2.resize(template, (img.shape[1], img.shape[0]), None, interpolation=cv2.INTER_LANCZOS4)
    new = cv2.resize(new, (img.shape[1], img.shape[0]), None, interpolation=cv2.INTER_LANCZOS4)
    hR, wR = img.shape[:-1]

    template = cv2.copyMakeBorder(template,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=(255, 255, 255))
    img = cv2.copyMakeBorder(img,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=(255, 255, 255))
    new = cv2.copyMakeBorder(new,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=(255, 255, 255))

    # Load the image pair.
    image0_bgra, image0, inp0, scales0, h0, w0 = process_image(
        template, device, opt.resize, 0, opt.resize_float)
    image1_bgra, image1, inp1, scales1, h1, w1 = process_image(
        img, device, opt.resize, 0, opt.resize_float)
    image2_bgra, image2, inp2, scales2, h2, w2 = process_image(
        new, device, opt.resize, 0, opt.resize_float)

    # Perform the matching.
    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    ### keypoint+matching visualization
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
    out = make_matching_plot_fast(
            image0, image1, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=False, small_text=small_text, opencv_display=False)
    # out = make_matching_plot_fast(
    #         image0, image1, kpts0, kpts1, mkpts0, mkpts1, color, text,
    #         path=None, show_keypoints=True, small_text=small_text, opencv_display=True)
    ### keypoint+matching visualization

    mkpts0 = mkpts0.astype(np.int32)
    mkpts1 = mkpts1.astype(np.int32)

    ## only do this for rectangle and square signals
    if category in ['squares', 'vertical_rect', 'horizontal_rect'] or len(mkpts0) < 4 or len(mkpts1) < 4:
        alpha_template = image0_bgra[:,:,-1]
        _, alpha_template = cv2.threshold(alpha_template,127,255,cv2.THRESH_BINARY)
        alpha_template = alpha_template.astype(np.uint8)
        alpha_img = image1_bgra[:,:,-1]
        _, alpha_img = cv2.threshold(alpha_img,127,255,cv2.THRESH_BINARY)
        alpha_img = alpha_img.astype(np.uint8)
        
        cnts1, _ = cv2.findContours(alpha_template, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cntsSorted1 = sorted(cnts1, key=lambda x: cv2.contourArea(x))[0]
        cnts2, _ = cv2.findContours(alpha_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cntsSorted2 = sorted(cnts2, key=lambda x: cv2.contourArea(x))[0]

        rect1 = cv2.minAreaRect(cntsSorted1)
        box1 = cv2.boxPoints(rect1)
        box1 = np.int0(box1)
        box1 = order_points(box1)
        box1 = box1.clip(min=0)

        rect2 = cv2.minAreaRect(cntsSorted2)
        box2 = cv2.boxPoints(rect2)
        box2 = np.int0(box2)
        box2 = order_points(box2)
        box2 = box2.clip(min=0)
        mkpts0 = np.append(mkpts0, box1, axis=0)
        mkpts1 = np.append(mkpts1, box2, axis=0)

    h, status = cv2.findHomography(mkpts0, mkpts1)
    aligned_template = cv2.warpPerspective(image0_bgra, h, (image1_bgra.shape[1],image1_bgra.shape[0]))
    aligned_result = cv2.warpPerspective(image2_bgra, h, (image1_bgra.shape[1],image1_bgra.shape[0]))

    # new
    alpha_new = aligned_result[:,:,-1]
    _, alpha_new = cv2.threshold(alpha_new,127,255,cv2.THRESH_BINARY)
    alpha_new = alpha_new.astype(np.uint8)
    alpha_img = image1_bgra[:,:,-1]
    _, alpha_img = cv2.threshold(alpha_img,127,255,cv2.THRESH_BINARY)
    alpha_img = alpha_img.astype(np.uint8)
    
    cnts1, _ = cv2.findContours(alpha_new, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted1 = sorted(cnts1, key=lambda x: cv2.contourArea(x))[0]
    cnts2, _ = cv2.findContours(alpha_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted2 = sorted(cnts2, key=lambda x: cv2.contourArea(x))[0]

    x1,y1,w,h = cv2.boundingRect(cntsSorted1)
    pts_src = np.array([[x1, y1], [x1+w, y1], [x1+w, y1+h],[x1, y1+h]])

    x1,y1,w,h = cv2.boundingRect(cntsSorted2)
    pts_dst = np.array([[x1, y1], [x1+w, y1], [x1+w, y1+h],[x1, y1+h]])

    h, status = cv2.findHomography(pts_src, pts_dst)
    aligned_result = cv2.warpPerspective(aligned_result, h, (image1_bgra.shape[1],image1_bgra.shape[0]))

    # template
    alpha_template = aligned_template[:,:,-1]
    _, alpha_template = cv2.threshold(alpha_template,127,255,cv2.THRESH_BINARY)
    alpha_template = alpha_template.astype(np.uint8)
    alpha_img = image1_bgra[:,:,-1]
    _, alpha_img = cv2.threshold(alpha_img,127,255,cv2.THRESH_BINARY)
    alpha_img = alpha_img.astype(np.uint8)
    
    cnts1, _ = cv2.findContours(alpha_template, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted1 = sorted(cnts1, key=lambda x: cv2.contourArea(x))[0]
    cnts2, _ = cv2.findContours(alpha_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted2 = sorted(cnts2, key=lambda x: cv2.contourArea(x))[0]

    x1,y1,w,h = cv2.boundingRect(cntsSorted1)
    pts_src = np.array([[x1, y1], [x1+w, y1], [x1+w, y1+h],[x1, y1+h]])

    x1,y1,w,h = cv2.boundingRect(cntsSorted2)
    pts_dst = np.array([[x1, y1], [x1+w, y1], [x1+w, y1+h],[x1, y1+h]])

    h, status = cv2.findHomography(pts_src, pts_dst)
    aligned_template = cv2.warpPerspective(aligned_template, h, (image1_bgra.shape[1],image1_bgra.shape[0]))

    image0_bgra = image0_bgra.astype(np.uint8)
    image1_bgra = image1_bgra.astype(np.uint8)
    image2_bgra = image2_bgra.astype(np.uint8)
    aligned_template = aligned_template.astype(np.uint8)
    aligned_result = aligned_result.astype(np.uint8)

    template = template[pad:-pad, pad:-pad,:]
    new = new[pad:-pad, pad:-pad,:]
    img = img[pad:-pad, pad:-pad,:]
    aligned_template = aligned_template[pad:-pad, pad:-pad,:]
    aligned_result = aligned_result[pad:-pad, pad:-pad,:]
    image0_bgra = image0_bgra[pad:-pad, pad:-pad,:]
    image1_bgra = image1_bgra[pad:-pad, pad:-pad,:]
    image2_bgra = image2_bgra[pad:-pad, pad:-pad,:]

    w = wR // 2
    h = hR // 2
    image0_bgra = cv2.resize(image0_bgra, (w, h), None, interpolation=cv2.INTER_AREA)
    image1_bgra = cv2.resize(image1_bgra, (w, h), None, interpolation=cv2.INTER_AREA)
    image2_bgra = cv2.resize(image2_bgra, (w, h), None, interpolation=cv2.INTER_AREA)
    aligned_template = cv2.resize(aligned_template, (w, h), None, interpolation=cv2.INTER_AREA)
    aligned_result = cv2.resize(aligned_result, (w, h), None, interpolation=cv2.INTER_AREA)

    out2 = np.hstack((image0_bgra, image1_bgra, aligned_template, image2_bgra, aligned_result))

    # cv2.imshow('result', np.hstack((image0_bgra, image1_bgra, aligned_template, image2_bgra, aligned_result)))
    # key = cv2.waitKey(0)
    # if key == 27:
    #     cv2.destroyAllWindows()
    #     exit()
    # for i in range(3):
    #     print('category: ', category)
    return aligned_template, aligned_result, out, out2