import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from pycocotools.coco import COCO
import torchvision.transforms as T
import albumentations as A
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import cv2

class DFGTrafficSignDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, split_path, annFile, transforms=None):
        super(DFGTrafficSignDataset, self).__init__()
        self.img_dir = img_dir
        self.transforms = transforms
        self.toTensor = T.ToTensor()
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgIDs = open(split_path, 'r').readlines()
        self.imgIDs = [int(x.strip()) for x in self.imgIDs]
        self.coco = COCO(annFile)

        # self.classes = ['I-1', 'I-1.1', 'I-10', 'I-11', 'I-13', 'I-13.1', 'I-14', 'I-15', 'I-16', 'I-17', 'I-18', 
        #     'I-19', 'I-2', 'I-2.1', 'I-20', 'I-25', 'I-27', 'I-28', 'I-28.1', 'I-29', 'I-29.1', 'I-3', 
        #     'I-30', 'I-32', 'I-34', 'I-36', 'I-37', 'I-38', 'I-39-1', 'I-39-2', 'I-39-3', 'I-4', 'I-5', 
        #     'I-5.1', 'I-5.2', 'I-8', 'I-9', 'II-1', 'II-10.1', 'II-14', 'II-17', 'II-18', 'II-19-4', 'II-2', 
        #     'II-21', 'II-22', 'II-23', 'II-26', 'II-26.1', 'II-28', 'II-3', 'II-30-10', 'II-30-30', 'II-30-40', 
        #     'II-30-50', 'II-30-60', 'II-30-70', 'II-32', 'II-33', 'II-34', 'II-35', 'II-39', 'II-4', 'II-40', 
        #     'II-41', 'II-42', 'II-42.1', 'II-43', 'II-45', 'II-45.1', 'II-45.2', 'II-46', 'II-46.1', 'II-46.2', 
        #     'II-47', 'II-47.1', 'II-48', 'II-6', 'II-7', 'II-7.1', 'II-8', 'III-1', 'III-10', 'III-105', 
        #     'III-105.1', 'III-105.3', 'III-107-1', 'III-107-2', 'III-107.1-1', 'III-107.1-2', 'III-107.2-1', 
        #     'III-107.2-2', 'III-112', 'III-113', 'III-12', 'III-120', 'III-120-1', 'III-120.1', 'III-123', 
        #     'III-124', 'III-14', 'III-14.1', 'III-15', 'III-16', 'III-18-40', 'III-18-50', 'III-18-60', 
        #     'III-18-70', 'III-2', 'III-202-5', 'III-203-2', 'III-206-1', 'III-21', 'III-23', 'III-25', 
        #     'III-25.1', 'III-27', 'III-29-30', 'III-29-40', 'III-3', 'III-30-30', 'III-33', 'III-34', 
        #     'III-35', 'III-37', 'III-39', 'III-40', 'III-42', 'III-43', 'III-45', 'III-46', 'III-47', 'III-5', 
        #     'III-50', 'III-54', 'III-59', 'III-6', 'III-64', 'III-68', 'III-74', 'III-77', 'III-78', 'III-8-1', 
        #     'III-84', 'III-84-1', 'III-85-2', 'III-85-3', 'III-85.1', 'III-86-1', 'III-86-2', 'III-87', 
        #     'III-90', 'III-90.1', 'III-90.2', 'III-91', 'IV-1', 'IV-1.1', 'IV-10', 'IV-11', 'IV-12', 'IV-12.1', 
        #     'IV-13-1', 'IV-13-2', 'IV-13-3', 'IV-13-4', 'IV-13-5', 'IV-13-6', 'IV-13.1-2', 'IV-13.1-3', 
        #     'IV-13.1-4', 'IV-16', 'IV-17', 'IV-18', 'IV-2', 'IV-20-1', 'IV-3-1', 'IV-3-2', 'IV-3-4', 
        #     'IV-3-5', 'IV-5', 'IV-6', 'VI-2.1', 'VI-3-1', 'VI-3-2', 'VI-3.1-1', 'VI-3.1-2', 'VI-8', 'VII-4', 
        #     'VII-4-1', 'VII-4-2', 'VII-4.1-1', 'VII-4.3', 'VII-4.3-1', 'VII-4.3-2', 'VII-4.4-1', 'VII-4.4-2', 
        #     'X-1.1', 'X-1.2', 'X-4', 'X-6-3']

        # self.palette = [[168, 140, 21], [81, 147, 215], [120, 96, 195], [175, 140, 37], [17, 218, 132], 
        #     [38, 192, 53], [104, 231, 173], [48, 70, 71], [36, 82, 43], [129, 110, 69], [192, 26, 113], 
        #     [195, 233, 30], [131, 20, 232], [215, 54, 168], [199, 173, 54], [67, 233, 72], [24, 16, 15], 
        #     [75, 229, 231], [78, 59, 16], [53, 39, 149], [90, 161, 173], [102, 117, 98], [99, 104, 242], 
        #     [167, 190, 34], [38, 101, 7], [232, 20, 145], [75, 119, 122], [188, 46, 63], [202, 176, 116], 
        #     [114, 230, 238], [159, 36, 118], [43, 184, 119], [87, 59, 39], [224, 192, 67], [40, 7, 52], 
        #     [206, 62, 109], [28, 223, 54], [218, 68, 106], [89, 149, 188], [203, 206, 76], [51, 67, 153], 
        #     [210, 42, 71], [144, 200, 160], [142, 134, 58], [167, 140, 3], [211, 205, 205], [66, 139, 99], 
        #     [196, 200, 1], [55, 30, 107], [10, 77, 169], [248, 15, 56], [191, 196, 121], [218, 186, 104], 
        #     [13, 108, 146], [195, 0, 192], [79, 112, 5], [121, 166, 252], [195, 160, 86], [133, 19, 123], 
        #     [118, 44, 247], [122, 246, 1], [15, 41, 5], [155, 97, 98], [170, 192, 92], [21, 108, 17], 
        #     [242, 108, 248], [181, 45, 201], [99, 105, 246], [148, 27, 94], [14, 124, 156], [120, 227, 254], 
        #     [112, 168, 193], [105, 78, 238], [79, 69, 238], [217, 50, 146], [216, 245, 67], [53, 30, 230], 
        #     [121, 115, 206], [240, 86, 183], [210, 68, 86], [209, 127, 152], [112, 17, 5], [205, 43, 101], 
        #     [99, 246, 56], [129, 138, 186], [102, 71, 122], [224, 66, 160], [166, 163, 44], [72, 34, 8], 
        #     [36, 177, 188], [209, 215, 193], [203, 39, 222], [228, 112, 135], [161, 48, 83], [157, 47, 237], 
        #     [65, 110, 225], [117, 154, 30], [10, 209, 43], [220, 79, 65], [245, 89, 86], [117, 107, 174], 
        #     [50, 145, 209], [140, 182, 81], [47, 41, 116], [136, 32, 69], [162, 47, 124], [209, 158, 44], 
        #     [196, 25, 71], [187, 252, 237], [106, 186, 129], [60, 173, 80], [34, 44, 75], [149, 129, 11], 
        #     [136, 236, 241], [16, 11, 156], [41, 171, 100], [36, 176, 106], [108, 172, 43], [45, 60, 178], 
        #     [182, 242, 180], [18, 71, 235], [239, 121, 221], [25, 56, 129], [246, 103, 226], [16, 145, 73], 
        #     [89, 100, 52], [222, 8, 105], [174, 55, 94], [90, 67, 214], [76, 254, 5], [159, 252, 236], 
        #     [252, 182, 43], [162, 232, 113], [129, 132, 39], [196, 199, 217], [71, 148, 37], [223, 232, 69], 
        #     [7, 216, 235], [34, 192, 17], [7, 59, 54], [27, 177, 173], [177, 202, 58], [38, 173, 86], 
        #     [67, 5, 153], [159, 167, 49], [2, 159, 252], [184, 21, 87], [128, 245, 239], [61, 70, 29], 
        #     [210, 123, 241], [95, 4, 56], [232, 143, 83], [135, 109, 31], [5, 11, 110], [116, 97, 231], 
        #     [139, 140, 118], [200, 252, 39], [81, 214, 84], [117, 212, 210], [20, 146, 32], [200, 44, 247], 
        #     [180, 77, 129], [211, 168, 115], [143, 210, 103], [94, 226, 239], [110, 113, 159], [83, 42, 209], 
        #     [5, 71, 113], [99, 70, 164], [106, 191, 226], [159, 69, 18], [9, 122, 230], [52, 148, 83], 
        #     [174, 68, 251], [185, 228, 192], [3, 244, 163], [245, 1, 34], [35, 128, 30], [63, 107, 158], 
        #     [17, 97, 219], [239, 104, 81], [184, 163, 86], [172, 87, 252], [133, 17, 95], [155, 119, 23], 
        #     [55, 251, 144], [61, 218, 185], [16, 86, 189], [171, 52, 75], [61, 74, 202], [133, 186, 75], 
        #     [159, 142, 43], [125, 209, 148], [36, 240, 218], [55, 195, 218], [169, 140, 26], [165, 43, 104], 
        #     [193, 249, 141], [95, 61, 113], [101, 206, 187]]

        self.classes = ['__background__', 'I-1', 'I-1.1', 'I-10', 'I-11', 'I-13', 'I-13.1', 'I-14', 'I-15', 'I-16', 'I-17', 'I-18', 'I-19', 
                'I-2', 'I-2.1', 'I-20', 'I-25', 'I-27', 'I-28', 'I-28.1', 'I-29', 'I-29.1', 'I-3', 'I-30', 'I-32', 
                'I-34', 'I-36', 'I-37', 'I-38', 'I-39-1', 'I-39-2', 'I-39-3', 'I-4', 'I-5', 'I-5.1', 'I-5.2', 
                'I-8', 'I-9', 'II-1', 'II-10.1', 'II-14', 'II-17', 'II-18', 'II-2', 'II-26', 'II-26.1', 'II-28', 
                'II-3', 'II-30-10', 'II-30-30', 'II-30-40', 'II-30-50', 'II-30-60', 'II-30-70', 'II-32', 'II-35', 
                'II-39', 'II-4', 'II-40', 'II-45', 'II-45.1', 'II-45.2', 'II-46', 'II-46.1', 'II-46.2', 'II-47', 
                'II-47.1', 'II-6', 'II-7', 'II-7.1', 'II-8', 'III-1', 'III-10', 'III-107-1', 'III-107-2', 'III-107.1-1', 
                'III-107.1-2', 'III-107.2-1', 'III-107.2-2', 'III-112', 'III-12', 'III-120.1', 'III-123', 'III-124', 
                'III-16', 'III-18-40', 'III-18-50', 'III-18-60', 'III-18-70', 'III-2', 'III-202-5', 'III-21', 'III-23', 
                'III-29-30', 'III-29-40', 'III-3', 'III-30-30', 'III-33', 'III-34', 'III-35', 'III-5', 'III-6', 'III-8-1', 
                'IV-12', 'IV-12.1', 'IV-13-5', 'IV-13-6', 'IV-13.1-2', 'IV-17', 'IV-18', 'IV-20-1', 'VI-2.1', 'VI-3-1', 
                'VI-3-2', 'VI-3.1-1', 'VI-3.1-2', 'X-4']

        self.palette = [[168, 140, 21], [81, 147, 215], [120, 96, 195], [175, 140, 37], [17, 218, 132], 
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
                [136, 236, 241], [16, 11, 156], [41, 171, 100], [36, 176, 106]]


    def __getitem__(self, idx):
        # load images and masks
        img_data = self.coco.loadImgs(self.imgIDs[idx])[0]
        img_path = '{}/{}'.format(self.img_dir, img_data['file_name'])
        annIds = self.coco.getAnnIds(imgIds=img_data['id'], iscrowd=None)
        anns = self.coco.loadAnns(annIds)

        img = Image.open(img_path).convert("RGB")
        # img = cv2.imread(img_path, 1)

        # masks = np.zeros((len(anns), img_data['height'],img_data['width']), dtype = np.uint8)
        masks = []
        bboxes = []
        labels = []

        for i in range(len(anns)):
            className = self.classes[anns[i]['category_id']]
            mask = np.zeros((img_data['height'],img_data['width']), dtype=np.uint8)
            bbox = np.array(anns[i]['bbox'], dtype=np.int32)
            pixel_value = self.classes.index(className)
            # mask = np.maximum(self.coco.annToMask(anns[i])*pixel_value, mask)
            # mask = np.maximum(self.coco.annToMask(anns[i])*1, mask)
            try:
                mask = self.coco.annToMask(anns[i])
            except:
                print(anns[i])
            
            xmin = max(bbox[0], 1)
            ymin = max(bbox[1], 1)
            xmax = min(bbox[0]+bbox[2], img_data['width']-1)
            ymax = min(bbox[1]+bbox[3], img_data['height']-1)
            # print('xmin: {}, ymin: {}, xmax: {}, ymax: {}'.format(xmin, ymin, xmax, ymax))
            # if xmax <= xmin or ymax <= ymin:
            #     print('1. xmin: {}, ymin: {}, xmax: {}, ymax: {} {} {}'.format(xmin, ymin, xmax, ymax, xmax <= xmin, ymax <= ymin))
            #     continue
            # else:
            # xmin = bbox[0]
            # ymin = bbox[1]
            # xmax = bbox[0]+bbox[2]
            # ymax = bbox[1]+bbox[3]
            masks.append(mask)
            labels.append(pixel_value)
            bboxes.append([xmin, ymin, xmax, ymax])
            
        
        if self.transforms is not None:
            augmented = self.transforms(image=np.array(img), masks=masks, bboxes=bboxes, category_id=labels)
            img = augmented['image']
            masks = augmented['masks']
            bboxes = augmented['bboxes']

            # plot_image_anns(np.array(img), masks, bboxes)
            # plot_image_anns(aug_img, aug_masks, aug_boxes)

        img = self.toTensor(img)
        bboxes = torch.as_tensor(np.array(bboxes, dtype=np.float32), dtype=torch.float32)
        labels = torch.as_tensor(np.array(labels, dtype=np.int64), dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks, dtype=np.uint8), dtype=torch.uint8)

        image_id = torch.tensor([self.imgIDs[idx]])
        # try:
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        # except:
        #     print('bboxes: ', bboxes)
        #     print('Image ID: ', self.imgIDs[idx])
        iscrowd = torch.zeros((len(anns),), dtype=torch.int64)

        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        return img, target

    def __len__(self):
        return len(self.imgIDs)


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.PILToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def visualize_bbox(img, bbox, color=(255, 255, 0), thickness=2):  
    """Helper to add bboxes to images 
    Args:
        img : image as open-cv numpy array
        bbox : boxes as a list or numpy array in pascal_voc fromat [x_min, y_min, x_max, y_max]  
        color=(255, 255, 0): boxes color 
        thickness=2 : boxes line thickness
    """
    print(img.shape)
    print('bbox: ', bbox)
    x_min, y_min, x_max, y_max = bbox
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
    return img


def plot_image_anns(image, masks, boxes=None):
    """Helper to plot images with bboxes and masks 
    Args:
        image: image as open-cv numpy array, original and augmented
        masks: setof binary masks, original and augmented 
        bbox : boxes as a list or numpy array, original and augmented       
    """
    # glue binary masks together
    one_mask = np.zeros_like(masks[0])
    for i, mask in enumerate(masks):
        one_mask += (mask > 0).astype(np.uint8) * (11-i) # (11-i) so my inner artist is happy with the masks colors     
      
    if boxes is not None:
        for box in boxes:
            image = visualize_bbox(image, box)
            
    # for binary masks we get one channel and need to convert to RGB for visualization
    mask_rgb = label2rgb(one_mask, bg_label=0)            
        
    f, ax = plt.subplots(1, 2, figsize=(16, 16))             
    ax[0].imshow(image)
    ax[0].set_title('Original image')     
    ax[1].imshow(mask_rgb, interpolation='nearest')
    ax[1].set_title('Original mask')
    f.tight_layout()
    plt.show() 


def plot_image_aug(image, image_aug, masks, aug_masks, boxes, aug_boxes):
    """Helper to plot images with bboxes and masks and their augmented versions 
    Args:
        image, image_aug: image as open-cv numpy array, original and augmented
        masks, aug_masks:setof binary masks, original and augmented 
        bbox, aug_boxes : boxes as a list or numpy array, original and augmented       
    """
    # glue masks together
    one_mask = np.zeros_like(masks[0])
    for i, mask in enumerate(masks):
        one_mask += (mask > 0).astype(np.uint8) * (11-i)       
    
    one_aug_mask = np.zeros_like(aug_masks[0])
    for i, augmask in enumerate(aug_masks):
        one_aug_mask += (augmask > 0).astype(np.uint8) * (11-i) 
    
    for box in boxes:
        image = visualize_bbox(image, box)
    for augbox in aug_boxes:
        image_aug = visualize_bbox(image_aug, augbox)    
        
    # for binary masks we get one channel and need to convert to RGB for visualization
    mask_rgb = label2rgb(one_mask, bg_label=0)            
    mask_aug_rgb = label2rgb(one_aug_mask, bg_label=0) 
    
    f, ax = plt.subplots(2, 2, figsize=(16, 16))             
    ax[0, 0].imshow(img)
    ax[0, 0].set_title('Original image')        
    ax[0, 1].imshow(image_aug)
    ax[0, 1].set_title('Augmented image')     
    ax[1, 0].imshow(mask_rgb, interpolation='nearest')
    ax[1, 0].set_title('Original mask')
    ax[1, 1].imshow(mask_aug_rgb, interpolation='nearest')
    ax[1, 1].set_title('Augmented mask')
    f.tight_layout()
    plt.show() 

def main():
    imgDir = '../../dataset/DFG_traffic_sign_dataset/JPEGImages'
    split_path = '../../dataset/DFG_traffic_sign_dataset/splits/test.txt'
    annFile = '../../dataset/DFG_traffic_sign_dataset/DFG-tsd-aug-annot-json/test.json'

    train_transform = A.Compose(
        [
            A.LongestMaxSize(512, interpolation=1),
            A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.25),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.25),
            A.RandomBrightnessContrast(p=0.5),
            A.OneOf([
                        A.MotionBlur(p=0.2),
                        A.MedianBlur(blur_limit=3, p=0.1),
                        A.Blur(blur_limit=3, p=0.1),
                    ], p=0.5),                    
            A.OneOf([                        
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
            ], p=0.5),
            A.HueSaturationValue(p=0.3),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ],
        bbox_params={'format':'pascal_voc', 'min_area': 0, 'min_visibility': 0, 'label_fields': ['category_id']}, p=1
    )

    dataloader = DFGTrafficSignDataset(imgDir, split_path, annFile, train_transform)
    img, target = dataloader.__getitem__(13)
    print('img: ', img.shape)
    print('target: ', target)


if __name__ == '__main__':
    main()