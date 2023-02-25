import os, sys
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import cv2
## For visualizing results
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple
from tqdm import tqdm
from PIL import Image
import random
import mmcv
import os.path as osp

def main():
    img_dir = '/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/JPEGImages'
    annFile = '/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/DFG-tsd-annot-json/test.json'
    split_path = '/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/DFG-tsd-annot-json'

    classes = ['I-1', 'I-1.1', 'I-10', 'I-11', 'I-13', 'I-13.1', 'I-14', 'I-15', 'I-16', 'I-17', 'I-18', 
            'I-19', 'I-2', 'I-2.1', 'I-20', 'I-25', 'I-27', 'I-28', 'I-28.1', 'I-29', 'I-29.1', 'I-3', 
            'I-30', 'I-32', 'I-34', 'I-36', 'I-37', 'I-38', 'I-39-1', 'I-39-2', 'I-39-3', 'I-4', 'I-5', 
            'I-5.1', 'I-5.2', 'I-8', 'I-9', 'II-1', 'II-10.1', 'II-14', 'II-17', 'II-18', 'II-19-4', 'II-2', 
            'II-21', 'II-22', 'II-23', 'II-26', 'II-26.1', 'II-28', 'II-3', 'II-30-10', 'II-30-30', 'II-30-40', 
            'II-30-50', 'II-30-60', 'II-30-70', 'II-32', 'II-33', 'II-34', 'II-35', 'II-39', 'II-4', 'II-40', 
            'II-41', 'II-42', 'II-42.1', 'II-43', 'II-45', 'II-45.1', 'II-45.2', 'II-46', 'II-46.1', 'II-46.2', 
            'II-47', 'II-47.1', 'II-48', 'II-6', 'II-7', 'II-7.1', 'II-8', 'III-1', 'III-10', 'III-105', 
            'III-105.1', 'III-105.3', 'III-107-1', 'III-107-2', 'III-107.1-1', 'III-107.1-2', 'III-107.2-1', 
            'III-107.2-2', 'III-112', 'III-113', 'III-12', 'III-120', 'III-120-1', 'III-120.1', 'III-123', 
            'III-124', 'III-14', 'III-14.1', 'III-15', 'III-16', 'III-18-40', 'III-18-50', 'III-18-60', 
            'III-18-70', 'III-2', 'III-202-5', 'III-203-2', 'III-206-1', 'III-21', 'III-23', 'III-25', 
            'III-25.1', 'III-27', 'III-29-30', 'III-29-40', 'III-3', 'III-30-30', 'III-33', 'III-34', 
            'III-35', 'III-37', 'III-39', 'III-40', 'III-42', 'III-43', 'III-45', 'III-46', 'III-47', 'III-5', 
            'III-50', 'III-54', 'III-59', 'III-6', 'III-64', 'III-68', 'III-74', 'III-77', 'III-78', 'III-8-1', 
            'III-84', 'III-84-1', 'III-85-2', 'III-85-3', 'III-85.1', 'III-86-1', 'III-86-2', 'III-87', 
            'III-90', 'III-90.1', 'III-90.2', 'III-91', 'IV-1', 'IV-1.1', 'IV-10', 'IV-11', 'IV-12', 'IV-12.1', 
            'IV-13-1', 'IV-13-2', 'IV-13-3', 'IV-13-4', 'IV-13-5', 'IV-13-6', 'IV-13.1-2', 'IV-13.1-3', 
            'IV-13.1-4', 'IV-16', 'IV-17', 'IV-18', 'IV-2', 'IV-20-1', 'IV-3-1', 'IV-3-2', 'IV-3-4', 
            'IV-3-5', 'IV-5', 'IV-6', 'VI-2.1', 'VI-3-1', 'VI-3-2', 'VI-3.1-1', 'VI-3.1-2', 'VI-8', 'VII-4', 
            'VII-4-1', 'VII-4-2', 'VII-4.1-1', 'VII-4.3', 'VII-4.3-1', 'VII-4.3-2', 'VII-4.4-1', 'VII-4.4-2', 
            'X-1.1', 'X-1.2', 'X-4', 'X-6-3']
    
    coco = COCO(annFile)

    imgIDs = coco.getImgIds()

    problems = 0

    filename_list = []

    for idx in tqdm(range(len(imgIDs))):
        # print(imgIDs[idx])
        # img_data = coco.loadImgs(0)[0]
        img_data = coco.loadImgs(imgIDs[idx])[0]
        img_path = '{}/{}'.format(img_dir, img_data['file_name'])
        annIds = coco.getAnnIds(imgIds=img_data['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)

        img = Image.open(img_path).convert("RGB")

        true_count, false_count = 0, 0

        for i in range(len(anns)):
            ignore = bool(anns[i]["ignore"])
            area = anns[i]["area"]
            if area > 900:
                true_count += 1
            else:
                # print(anns[i])
                false_count += 1
                continue
            
            label = anns[i]['category_id']
            if label == 0:
                print(anns[i])

            # mask = coco.annToMask(anns[i])*255
            # print(np.unique(mask))
            # print(mask.shape)

            # cv2.imshow(str(imgIDs[idx]), mask)
            # key = cv2.waitKey(0)
            # if key == 27:
            #     cv2.destroyAllWindows(0)
            #     exit()

            bbox = np.array(anns[i]['bbox'], dtype=np.int32)
            xmin = max(bbox[0], 0)
            ymin = max(bbox[1], 0)
            xmax = min(bbox[0]+bbox[2], img_data['width'])
            ymax = min(bbox[1]+bbox[3], img_data['height'])
            if ymax <= ymin+5 or xmax <= xmin+5:
                print(anns[i])

        # exit()
        if true_count == 0:
            problems += 1
            # print('True: {}, false: {}'.format(true_count, false_count))
        else:
            filename_list.append(imgIDs[idx])
        
    print('Problems: ', problems)

    # with open(osp.join(split_path, 'test.txt'), 'w+') as f:
    #     f.writelines(str(line) + '\n' for line in filename_list)

    # random.shuffle(filename_list)
    # random.shuffle(filename_list)
    # # select first 80% as train set
    # train_length = int(len(filename_list)*0.8)
    # with open(osp.join(split_path, 'train.txt'), 'w') as f:
    #     f.writelines(str(line) + '\n' for line in filename_list[:train_length])
    # with open(osp.join(split_path, 'val.txt'), 'w') as f:
    #     # select last 20% as val set
    #     f.writelines(str(line) + '\n' for line in filename_list[train_length:])
                
    

if __name__ == '__main__':
    main()