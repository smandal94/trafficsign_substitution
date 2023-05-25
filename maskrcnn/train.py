import os, sys
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
import albumentations as A

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.optim.lr_scheduler import StepLR
import src.utils as utils
from src.engine import train_one_epoch, evaluate

from src.dataset import DFGTrafficSignDataset
from src.model import get_instance_segmentation_model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    imgDir = '/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/JPEGImages'
    train_annFile = 'examples/arbitrary_removed/train.json'
    train_split = 'examples/arbitrary_removed/train.txt'
    val_split = 'examples/arbitrary_removed/val.txt'
    batch_size = 8
    num_workers = 16
    n_classes = 117
    num_epochs = 100
    imgsz = 800
    save_path = './checkpoints'

    train_transform = A.Compose(
        [
            A.LongestMaxSize(imgsz, interpolation=1),
            A.PadIfNeeded(min_height=imgsz, min_width=imgsz, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.25),
            A.RandomBrightnessContrast(p=0.5),
            A.OneOf([
                        A.MotionBlur(p=0.2),
                        A.MedianBlur(blur_limit=3, p=0.1),
                        A.Blur(blur_limit=3, p=0.1),
                    ], p=0.5),                    
            A.OneOf([                        
                A.Sharpen(),
                A.Emboss()
            ], p=0.5),
            A.HueSaturationValue(p=0.3),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ],
        bbox_params={'format':'pascal_voc', 'min_area': 0, 'min_visibility': 0, 'label_fields': ['category_id']}, p=1
    )
    train_dataset = DFGTrafficSignDataset(imgDir, train_split, train_annFile, train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                            collate_fn=utils.collate_fn)

    val_transform = A.Compose(
        [
            A.LongestMaxSize(imgsz, interpolation=1),
            A.PadIfNeeded(min_height=imgsz, min_width=imgsz, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1)
        ],
        bbox_params={'format':'pascal_voc', 'min_area': 0, 'min_visibility': 0, 'label_fields': ['category_id']}, p=1
    )
    val_dataset = DFGTrafficSignDataset(imgDir, val_split, train_annFile, val_transform)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size//2, shuffle=False, num_workers=0,
                                            collate_fn=utils.collate_fn)
    
    
    model = get_instance_segmentation_model(n_classes).cuda()
    print('Model: ', model)
    print('Params: ', count_parameters(model))

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0001)

    # and a learning rate scheduler which decreases the learning rate by
    # 5x every 30 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=30,
                                                gamma=0.05)

    
    for epoch in range(num_epochs):
        # train for one epoch, printing every 50 iterations
        train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=50)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the val dataset
        evaluate(model, val_data_loader, device=device)

        torch.save(model.state_dict(), '{}/maskrcnn_epoch{:03d}.pth'.format(save_path, epoch))

if __name__ == '__main__':
    main()