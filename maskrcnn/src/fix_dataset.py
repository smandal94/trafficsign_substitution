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

def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

def main():
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
    print('Number of classes: ', len(classes))
    palette = [[168, 140, 21], [81, 147, 215], [120, 96, 195], [175, 140, 37], [17, 218, 132], 
            [38, 192, 53], [104, 231, 173], [48, 70, 71], [36, 82, 43], [129, 110, 69], [192, 26, 113], 
            [195, 233, 30], [131, 20, 232], [215, 54, 168], [199, 173, 54], [67, 233, 72], [24, 16, 15], 
            [75, 229, 231], [78, 59, 16], [53, 39, 149], [90, 161, 173], [102, 117, 98], [99, 104, 242], 
            [167, 190, 34], [38, 101, 7], [232, 20, 145], [75, 119, 122], [188, 46, 63], [202, 176, 116], 
            [114, 230, 238], [159, 36, 118], [43, 184, 119], [87, 59, 39], [224, 192, 67], [40, 7, 52], 
            [206, 62, 109], [28, 223, 54], [218, 68, 106], [89, 149, 188], [203, 206, 76], [51, 67, 153], 
            [210, 42, 71], [144, 200, 160], [142, 134, 58], [167, 140, 3], [211, 205, 205], [66, 139, 99], 
            [196, 200, 1], [55, 30, 107], [10, 77, 169], [248, 15, 56], [191, 196, 121], [218, 186, 104], 
            [13, 108, 146], [195, 0, 192], [79, 112, 5], [121, 166, 252], [195, 160, 86], [133, 19, 123], 
            [118, 44, 247], [122, 246, 1], [15, 41, 5], [155, 97, 98], [170, 192, 92], [21, 108, 17], 
            [242, 108, 248], [181, 45, 201], [99, 105, 246], [148, 27, 94], [14, 124, 156], [120, 227, 254], 
            [112, 168, 193], [105, 78, 238], [79, 69, 238], [217, 50, 146], [216, 245, 67], [53, 30, 230], 
            [121, 115, 206], [240, 86, 183], [210, 68, 86], [209, 127, 152], [112, 17, 5], [205, 43, 101], 
            [99, 246, 56], [129, 138, 186], [102, 71, 122], [224, 66, 160], [166, 163, 44], [72, 34, 8], 
            [36, 177, 188], [209, 215, 193], [203, 39, 222], [228, 112, 135], [161, 48, 83], [157, 47, 237], 
            [65, 110, 225], [117, 154, 30], [10, 209, 43], [220, 79, 65], [245, 89, 86], [117, 107, 174], 
            [50, 145, 209], [140, 182, 81], [47, 41, 116], [136, 32, 69], [162, 47, 124], [209, 158, 44], 
            [196, 25, 71], [187, 252, 237], [106, 186, 129], [60, 173, 80], [34, 44, 75], [149, 129, 11], 
            [136, 236, 241], [16, 11, 156], [41, 171, 100], [36, 176, 106], [108, 172, 43], [45, 60, 178], 
            [182, 242, 180], [18, 71, 235], [239, 121, 221], [25, 56, 129], [246, 103, 226], [16, 145, 73], 
            [89, 100, 52], [222, 8, 105], [174, 55, 94], [90, 67, 214], [76, 254, 5], [159, 252, 236], 
            [252, 182, 43], [162, 232, 113], [129, 132, 39], [196, 199, 217], [71, 148, 37], [223, 232, 69], 
            [7, 216, 235], [34, 192, 17], [7, 59, 54], [27, 177, 173], [177, 202, 58], [38, 173, 86], 
            [67, 5, 153], [159, 167, 49], [2, 159, 252], [184, 21, 87], [128, 245, 239], [61, 70, 29], 
            [210, 123, 241], [95, 4, 56], [232, 143, 83], [135, 109, 31], [5, 11, 110], [116, 97, 231], 
            [139, 140, 118], [200, 252, 39], [81, 214, 84], [117, 212, 210], [20, 146, 32], [200, 44, 247], 
            [180, 77, 129], [211, 168, 115], [143, 210, 103], [94, 226, 239], [110, 113, 159], [83, 42, 209], 
            [5, 71, 113], [99, 70, 164], [106, 191, 226], [159, 69, 18], [9, 122, 230], [52, 148, 83], 
            [174, 68, 251], [185, 228, 192], [3, 244, 163], [245, 1, 34], [35, 128, 30], [63, 107, 158], 
            [17, 97, 219], [239, 104, 81], [184, 163, 86], [172, 87, 252], [133, 17, 95], [155, 119, 23], 
            [55, 251, 144], [61, 218, 185], [16, 86, 189], [171, 52, 75], [61, 74, 202], [133, 186, 75], 
            [159, 142, 43], [125, 209, 148], [36, 240, 218], [55, 195, 218], [169, 140, 26], [165, 43, 104], 
            [193, 249, 141], [95, 61, 113], [101, 206, 187]]
    print('Number of colors: ', len(palette))

    annFile='../../../dataset/DFG_traffic_sign_dataset/DFG-tsd-aug-annot-json/test.json'
    print(os.path.exists(annFile))

    # initialize the COCO api for instance annotations
    coco=COCO(annFile)

    # display COCO categories and supercategories
    catIDs = coco.getCatIds()
    cats = coco.loadCats(catIDs)

    print(len(classes),'COCO categories: \n{}\n'.format(' '.join(classes)))
    print(classes)

    imgIds = coco.getImgIds()

    dataDir = '../../../dataset/DFG_traffic_sign_dataset/JPEGImages'
    
    filename_list = []
    for idx in tqdm(range(len(imgIds))):
        # try:
        img = coco.loadImgs(imgIds[idx])[0]
        # except:
        #     print("Id {} does not exist!".format(idx))
        #     continue
        img_fname = '{}/{}'.format(dataDir, img['file_name'])
        sname = img_fname.replace('JPEGImages', 'labels').replace('.jpg', '.png')
        if os.path.exists(sname):
            continue
        # I = cv2.imread(img_fname, 1)
        # viz = cv2.resize(I, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)

        # create mask
        error = -1
        lbls = 0
        mask = np.zeros((img['height'],img['width']), dtype=np.int32)
        if len(anns) == 0:
            error = 1
        else:
            for i in range(len(anns)):
                className = classes[anns[i]['category_id']]
                pixel_value = classes.index(className)+1
                mask = np.maximum(coco.annToMask(anns[i])*pixel_value, mask)
                bbox = np.array(anns[i]['bbox'], dtype=np.int32)
                xmin = max(bbox[0], 0)
                ymin = max(bbox[1], 0)
                xmax = min(bbox[0]+bbox[2], img['width'])
                ymax = min(bbox[1]+bbox[3], img['height'])
                if ymax <= ymin+5 or xmax <= xmin+5:
                    error = 1
                else:
                    lbls += 1


            # print(tuple(palette[anns[i]['category_id']]))
            # I = cv2.rectangle(I, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), tuple(palette[anns[i]['category_id']]), 2)

        if error == 1:
            continue
        elif lbls == 0:
            continue
        else:
            print('error: {}, lbls: {}'.format(error, lbls))
            filename_list.append(imgIds[idx])

        # print('Unique pixel values in the mask are:', np.unique(mask))
        
        # seg_img = Image.fromarray(mask).convert('P')
        # seg_img.putpalette(np.array(palette, dtype=np.uint8))
        # print('sname: ', sname)
        # seg_img.save(sname)
        # cv2.imwrite(sname, mask)
        
        # plt.imshow(mask)
        # plt.show()
        # cv2.imshow('image', I)
        # print('img shape: ', I.shape)
        # # cv2.imshow('mask', mask)
        # key = cv2.waitKey(0)
        # if key == 27:
        #     cv2.destroyAllWindows()
        #     exit()
    # print('error: ', error)
    
    # split train/val set randomly
    data_root = '../../../dataset/DFG_traffic_sign_dataset'
    split_dir = 'splits'
    img_dir='JPEGImages'
    ann_dir='labels'
    mmcv.mkdir_or_exist(osp.join(data_root, split_dir))
    # filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
    #     osp.join(data_root, ann_dir), suffix='.png')]
    random.shuffle(filename_list)
    random.shuffle(filename_list)

    # # select first 80% as train set
    # train_length = int(len(filename_list)*0.8)
    # with open(osp.join(data_root, split_dir, 'train.txt'), 'w') as f:
    #     f.writelines(str(line) + '\n' for line in filename_list[:train_length])
    # with open(osp.join(data_root, split_dir, 'val.txt'), 'w') as f:
    #     # select last 20% as val set
    #     f.writelines(str(line) + '\n' for line in filename_list[train_length:])

    with open(osp.join(data_root, split_dir, 'test.txt'), 'w') as f:
        f.writelines(str(line) + '\n' for line in filename_list)


if __name__ == '__main__':
    main()
