import os, sys
import glob
import numpy as np
import cv2

import numpy as np
import copy
from scipy.spatial import distance
from shapely.geometry import Polygon

import logging
import shutil
import warnings
from typing import Dict, List, Tuple, cast

import kornia
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

# import theseus as th
# from theseus.core.cost_function import AutogradMode, ErrFnType
# from theseus.third_party.utils import grid_sample

if not sys.warnoptions:
    warnings.filterwarnings("ignore", category=UserWarning)

# BACKWARD_MODE = {
#     "implicit": th.BackwardMode.IMPLICIT,
#     "full": th.BackwardMode.FULL,
#     "truncated": th.BackwardMode.TRUNCATED,
# }


orb = cv2.ORB_create(
    nfeatures=10000,
    scaleFactor=1.2,
    scoreType=cv2.ORB_HARRIS_SCORE)
LOWES_RATIO = 0.7
MIN_MATCHES = 50
index_params = dict(
    algorithm = 6, # FLANN_INDEX_LSH
    table_number = 6,
    key_size = 10,
    multi_probe_level = 2)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(
    index_params,
    search_params)

class FeatureExtraction:
    def __init__(self, img):
        self.img = copy.copy(img)
        self.gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.kps, self.des = orb.detectAndCompute(self.gray_img, None)
        self.matched_pts = []


def feature_matching(features0, features1):
    matches = [] # good matches as per Lowe's ratio test
    if(features0.des is not None and len(features0.des) > 2):
        all_matches = flann.knnMatch(features0.des, features1.des, k=2)
        try:
            for m,n in all_matches:
                if m.distance < LOWES_RATIO * n.distance:
                    matches.append(m)
        except ValueError:
            pass
        if(len(matches) > MIN_MATCHES):    
            features0.matched_pts = np.float32([ features0.kps[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
            features1.matched_pts = np.float32([ features1.kps[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    return matches

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

def align(template, img, new):
    new = cv2.resize(new, (template.shape[1], template.shape[0]), None, interpolation=cv2.INTER_AREA)
    template = cv2.copyMakeBorder(template,50,50,50,50,cv2.BORDER_CONSTANT,value=(0, 0, 0))
    img = cv2.copyMakeBorder(img,50,50,50,50,cv2.BORDER_CONSTANT,value=(0, 0, 0))
    new = cv2.copyMakeBorder(new,50,50,50,50,cv2.BORDER_CONSTANT,value=(0, 0, 0))
    h, w = img.shape[:-1]

    alpha_template = template[:,:,-1]
    alpha_img = img[:,:,-1]
    alpha_new = new[:,:,-1]
    
    cnts1, h1 = cv2.findContours(alpha_template, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted1 = sorted(cnts1, key=lambda x: cv2.contourArea(x))[0]
    cnts2, h2 = cv2.findContours(alpha_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

    features0 = FeatureExtraction(template)
    features1 = FeatureExtraction(img)

    try:
        matches = feature_matching(features0, features1)
        matched_image = cv2.drawMatches(template, features0.kps, img, features1.kps, matches, None, flags=2)
        cv2.imshow("Matched Keypoints", matched_image)
    except:
        matches = []

    ptsA = np.zeros((len(matches)+4, 2), dtype="float")
    ptsB = np.zeros((len(matches)+4, 2), dtype="float")
    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = features0.kps[m.queryIdx].pt
        ptsB[i] = features1.kps[m.trainIdx].pt
    
    for i in range(len(matches), len(matches)+4):
        ptsA[i] = box1[i-len(matches)]
        ptsB[i] = box2[i-len(matches)]

    H, _ = cv2.findHomography( ptsA, ptsB, cv2.RANSAC, 5.0)

    h, w, c = img.shape
    aligned = cv2.warpPerspective(template, H, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    aligned2 = cv2.warpPerspective(new, H, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    cv2.imshow('org', img)
    cv2.imshow('template', template)
    cv2.imshow('new', new)
    cv2.imshow('T2O', aligned)
    cv2.imshow('N2O', aligned2)

    template = template[50:-50, 50:-50,:]
    new = new[50:-50, 50:-50,:]
    img = img[50:-50, 50:-50,:]
    aligned_template = aligned[50:-50, 50:-50,:]
    aligned_result = aligned2[50:-50, 50:-50,:]

    # cv2.imshow('target->align', np.hstack((img, aligned_template, aligned_result)))
    # cv2.imshow('target', img)
    # cv2.imshow('org', img)
    # cv2.imshow('template', template)
    # cv2.imshow('new', new)
    # cv2.imshow('T2O', aligned)
    # cv2.imshow('N2O', aligned2)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
        exit()
    else:
        cv2.destroyAllWindows()

    exit()

    return aligned_template, aligned_result

def main1():
    files = sorted(glob.glob('/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/crops/I-2.1/*.png'))

    # template = cv2.imread('/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/templates/I-16.png', -1)

    new = cv2.imread('/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/templates/I-16.png', -1)

    for f in files:
        template = cv2.imread('/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/templates/I-2.1.png', -1)
        img = cv2.imread(f, -1)
        new = cv2.resize(new, (template.shape[1], template.shape[0]), None, interpolation=cv2.INTER_LINEAR)
        template = cv2.copyMakeBorder(template,50,50,50,50,cv2.BORDER_CONSTANT,value=(0, 0, 0))
        img = cv2.copyMakeBorder(img,50,50,50,50,cv2.BORDER_CONSTANT,value=(0, 0, 0))
        new = cv2.copyMakeBorder(new,50,50,50,50,cv2.BORDER_CONSTANT,value=(0, 0, 0))
        h, w = img.shape[:-1]
        # template = cv2.resize(template, (w, h), None, interpolation=cv2.INTER_AREA)

        alpha_template = template[:,:,-1]
        alpha_img = img[:,:,-1]
        alpha_new = new[:,:,-1]
        
        cnts1, h1 = cv2.findContours(alpha_template, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cntsSorted1 = sorted(cnts1, key=lambda x: cv2.contourArea(x))[0]
        # hull1 = cv2.convexHull(cntsSorted1, False)

        cnts2, h2 = cv2.findContours(alpha_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cntsSorted2 = sorted(cnts2, key=lambda x: cv2.contourArea(x))[0]
        # hull2 = cv2.convexHull(cntsSorted2, False)

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

        cv2.drawContours(template, [box1], 0, (0, 255, 0), 2)
        # cv2.drawContours(template, [hull1], 0, (0, 255, 255), 2)
        cv2.drawContours(img, [box2], 0, (0, 255, 0), 2)
        # cv2.drawContours(img, [hull2], 0, (0, 255, 255), 2)

        cv2.imshow('a1', alpha_template)
        cv2.imshow('a2', alpha_img)
        cv2.imshow('a3', alpha_new)


        features0 = FeatureExtraction(template)
        features1 = FeatureExtraction(img)

        try:
            matches = feature_matching(features0, features1)
            matched_image = cv2.drawMatches(template, features0.kps, img, features1.kps, matches, None, flags=2)
            cv2.imshow("Matched Keypoints", matched_image)
        except:
            matches = []
        # matches = []
        ptsA = np.zeros((len(matches)+4, 2), dtype="float")
        ptsB = np.zeros((len(matches)+4, 2), dtype="float")
        # loop over the top matches
        for (i, m) in enumerate(matches):
            # indicate that the two keypoints in the respective images
            # map to each other
            ptsA[i] = features0.kps[m.queryIdx].pt
            ptsB[i] = features1.kps[m.trainIdx].pt
        
        for i in range(len(matches), len(matches)+4):
            ptsA[i] = box1[i-len(matches)]
            ptsB[i] = box2[i-len(matches)]

        H, _ = cv2.findHomography( ptsA, ptsB, cv2.RANSAC, 5.0)

        h, w, c = img.shape
        aligned = cv2.warpPerspective(template, H, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        aligned2 = cv2.warpPerspective(new, H, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        template = template[50:-50, 50:-50,:]
        new = new[50:-50, 50:-50,:]
        img = img[50:-50, 50:-50,:]
        aligned = aligned[50:-50, 50:-50,:]
        aligned2 = aligned2[50:-50, 50:-50,:]

        cv2.imshow('original', template)
        cv2.imshow('new', new)
        cv2.imshow('target->align', np.hstack((img, aligned, aligned2)))
        # cv2.imshow('target', img)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            exit()
        else:
            cv2.destroyAllWindows()


# def torch2cv2(img):
#     out = (img.permute(1, 2, 0) * 255.0).data.cpu().numpy().astype(np.uint8)[:, :, ::-1]
#     out = np.ascontiguousarray(out)
#     return out

# def warp_perspective_norm(H, img):
#     height, width = img.shape[-2:]
#     grid = kornia.utils.create_meshgrid(
#         height, width, normalized_coordinates=True, device=H.device
#     )
#     Hinv = torch.inverse(H)
#     warped_grid = kornia.geometry.transform.homography_warper.warp_grid(grid, Hinv)
#     # Using custom implementation, above will throw error with outer loop optim.
#     img2 = grid_sample(img, warped_grid)
#     return img2

# # loss is difference between warped and target image
# def homography_error_fn(optim_vars: Tuple[th.Manifold], aux_vars: Tuple[th.Variable]):
#     H8_1_2 = optim_vars[0].tensor.reshape(-1, 8)
#     # Force the last element H[2,2] to be 1.
#     H_1_2 = torch.cat([H8_1_2, H8_1_2.new_ones(H8_1_2.shape[0], 1)], dim=-1)  # type: ignore
#     img1: th.Variable = aux_vars[0]
#     img2: th.Variable = aux_vars[-1]
#     img1_dst = warp_perspective_norm(H_1_2.reshape(-1, 3, 3), img1.tensor)
#     loss = torch.nn.functional.mse_loss(img1_dst, img2.tensor, reduction="none")
#     ones = warp_perspective_norm(
#         H_1_2.data.reshape(-1, 3, 3), torch.ones_like(img1.tensor)
#     )
#     mask = ones > 0.9
#     loss = loss.view(loss.shape[0], -1)
#     mask = mask.view(loss.shape[0], -1)
#     loss = (loss * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)
#     return loss

# class SimpleCNN(nn.Module):
#     def __init__(self, D=32):
#         super(SimpleCNN, self).__init__()
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(3, D, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(D, 3, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(D)

#     def forward(self, img):
#         x = self.relu(self.bn1(self.conv1(img)))
#         return self.conv2(x)

# def main2():
#     files = sorted(glob.glob('/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/crops/I-32/*.png'))

#     # template = cv2.imread('/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/templates/I-16.png', -1)

#     new1 = cv2.imread('/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/templates/I-25.png', -1)

#     imgH, imgW = 96, 96
#     outer_lr = 1e-4
#     batch_size = 1
#     max_iterations = 50
#     step_size = 0.1
#     autograd_mode = AutogradMode.VMAP
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     # A simple 2-layer CNN network that maintains the original image size.
#     cnn_model = torch.load('./checkpoints/theseus_homography.ckpt')['cnn_model']
#     cnn_model.to(device)
#     print(cnn_model)

#     transform = transforms.Compose([
#         transforms.ToTensor()
#     ])

#     objective = th.Objective()

#     H8_init = torch.eye(3).reshape(1, 9)[:, :-1].repeat(1, 1)
#     H8_init = H8_init.to(device)
#     feats = torch.zeros((1, 3, imgH, imgW))
#     H8_1_2 = th.Vector(tensor=H8_init, name="H8_1_2")
#     feat1 = th.Variable(tensor=feats, name="feat1")
#     feat2 = th.Variable(tensor=feats, name="feat2")

#     # Set up inner loop optimization.
#     homography_cf = th.AutoDiffCostFunction(
#         optim_vars=[H8_1_2],
#         err_fn=cast(ErrFnType, homography_error_fn),
#         dim=1,
#         aux_vars=[feat1, feat2],
#         autograd_mode=autograd_mode,
#     )
#     objective.add(homography_cf)

#     # Regularization helps avoid crash with using implicit mode.
#     reg_w_value = 1e-2
#     reg_w = th.ScaleCostWeight(np.sqrt(reg_w_value))
#     reg_w.to(dtype=H8_init.dtype)
#     vals = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
#     H8_1_2_id = th.Vector(tensor=vals, name="identity")
#     reg_cf = th.Difference(
#         H8_1_2, target=H8_1_2_id, cost_weight=reg_w, name="reg_homography"
#     )
#     objective.add(reg_cf)

#     inner_optim = th.LevenbergMarquardt(
#         objective,
#         max_iterations=max_iterations,
#         step_size=step_size,
#     )
#     theseus_layer = th.TheseusLayer(inner_optim)
#     theseus_layer.to(device)

#     for f in files:
#         template = cv2.imread('/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/templates/I-32.png', -1)
#         img = cv2.imread(f, -1)
#         # new = cv2.resize(new1, (template.shape[1], template.shape[0]), None, interpolation=cv2.INTER_LINEAR)

#         h_org, w_org = img.shape[:2]

#         imgX = cv2.resize(img[:,:,:-1], (imgH, imgW), None, interpolation=cv2.INTER_AREA)
#         templateX = cv2.resize(template[:,:,:-1], (imgH, imgW), None, interpolation=cv2.INTER_AREA)
#         newX = cv2.resize(new1, (imgH, imgW), None, interpolation=cv2.INTER_AREA)

#         imgX = transform(imgX).unsqueeze(0).to(device)
#         templateX = transform(templateX).unsqueeze(0).to(device)
#         print(imgX.shape, templateX.shape)

#         feat1_tensor = cnn_model.forward(templateX)
#         feat2_tensor = cnn_model.forward(imgX)

#         inputs: Dict[str, torch.Tensor] = {
#                                             "H8_1_2": H8_init,
#                                             "feat1": feat1_tensor,
#                                             "feat2": feat2_tensor,
#                                         }
#         _, info = theseus_layer.forward(
#                                         inputs,
#                                         optimizer_kwargs={
#                                             "verbose": False,
#                                             "track_err_history": True,
#                                             "track_state_history": True,
#                                             "backward_mode": BACKWARD_MODE["implicit"],
#                                         },
#                                     )
#         H8_1_2_tensor = theseus_layer.objective.get_optim_var(
#                                                                 "H8_1_2"
#                                                             ).tensor.reshape(-1, 8)
#         H_1_2 = torch.cat(
#             [H8_1_2_tensor, H8_1_2_tensor.new_ones(H8_1_2_tensor.shape[0], 1)],
#             dim=-1,
#         )
#         print(H_1_2, H_1_2.shape)

#         H_1_2_mat = H_1_2[0].reshape(1, 3, 3)
#         img1 = templateX[0][None, ...]
#         img2 = imgX[0][None, ...]
#         img1_dsts = warp_perspective_norm(H_1_2_mat, img1)

#         img1 = torch2cv2(img1[0])[:,:,::-1]
#         img2 = torch2cv2(img2[0])[:,:,::-1]
#         img1_w = torch2cv2(img1_dsts[0])[:,:,::-1]
#         print(img1.shape, img2.shape, img1_w.shape)

#         H = H_1_2_mat.cpu().detach().numpy()[0]
#         print(H.shape, 'H')

#         # cv2.imshow('original', template)
#         # cv2.imshow('new', newX)
#         cv2.imshow('target->align', np.hstack((img1, img2, img1_w)))
#         key = cv2.waitKey(0)
#         if key == 27:
#             cv2.destroyAllWindows()
#             exit()
#         else:
#             cv2.destroyAllWindows()

if __name__ == '__main__':
    main1()