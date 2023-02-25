import os, sys

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
import albumentations as A

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler
import src.utils as utils
from src.engine import train_one_epoch, evaluate

from src.dataset import DFGTrafficSignDataset
from src.model import get_instance_segmentation_model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    imgDir = '../../dataset/DFG_traffic_sign_dataset/JPEGImages'
    train_annFile = '../../dataset/DFG_traffic_sign_dataset/DFG-tsd-annot-json/train.json'
    train_split = '../../dataset/DFG_traffic_sign_dataset/DFG-tsd-annot-json/train.txt'
    val_split = '../../dataset/DFG_traffic_sign_dataset/DFG-tsd-annot-json/val.txt'
    model_path = './checkpoints/maskrcnn_epoch021.pth'
    optim_path = './checkpoints/optim_epoch021.pth'
    lr_path = './checkpoints/lrscheduler_epoch005.pth'
    batch_size = 4
    num_workers = 12
    n_classes = 201
    num_epochs = 100
    start_epoch = 22
    imgsz = 800
    accum_iter = 8
    save_path = './checkpoints'

    train_transform = A.Compose(
        [
            A.LongestMaxSize(imgsz, interpolation=1),
            A.PadIfNeeded(min_height=imgsz, min_width=imgsz, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1),
            # Don't use ShiftScaleRotate
            # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.25),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.25),
            A.OneOf([
                        A.MotionBlur(),
                        A.MedianBlur(blur_limit=3),
                        A.Blur(blur_limit=3),
                    ], p=0.25),                    
            A.OneOf([                        
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
                A.HueSaturationValue(p=0.3),
                A.CLAHE(),    
                A.RandomGamma(),
            ], p=0.25),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ],
        bbox_params={'format':'pascal_voc', 'min_area': 0, 'min_visibility': 0, 'label_fields': ['category_id']}, p=1
    )
    train_dataset = DFGTrafficSignDataset(imgDir, train_split, train_annFile, train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                            collate_fn=utils.collate_fn)

    val_transform = A.Compose(
        [
            A.LongestMaxSize(imgsz, interpolation=1),
            A.PadIfNeeded(min_height=imgsz, min_width=imgsz, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1)
        ],
        bbox_params={'format':'pascal_voc', 'min_area': 0, 'min_visibility': 0, 'label_fields': ['category_id']}, p=1
    )
    val_dataset = DFGTrafficSignDataset(imgDir, val_split, train_annFile, val_transform)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size//2, shuffle=False, num_workers=num_workers//2,
                                            collate_fn=utils.collate_fn)
    
    
    model = get_instance_segmentation_model(n_classes).cuda()
    model.load_state_dict(torch.load(model_path))
    print('Model: ', model)
    print('Params: ', count_parameters(model))

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.0025, momentum=0.9, weight_decay=0.0001)
    optimizer = torch.optim.Adam(params, lr=0.0001)
    optimizer.load_state_dict(torch.load(optim_path))

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 50 epochs
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                             step_size=50,
    #                                             gamma=0.1)
    # lr_scheduler.load_state_dict(torch.load(lr_path))

    scaler = None

    for epoch in range(num_epochs):
        if epoch < start_epoch:
            continue
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=100, scaler=scaler, accum_iter=accum_iter)
        # update the learning rate
        # lr_scheduler.step()
        # evaluate on the val dataset
        evaluate(model, val_data_loader, device=device)

        torch.save(model.state_dict(), '{}/maskrcnn_epoch{:03d}.pth'.format(save_path, epoch))
        torch.save(optimizer.state_dict(), '{}/optim_epoch{:03d}.pth'.format(save_path, epoch))
        # torch.save(lr_scheduler.state_dict(), '{}/lrscheduler_epoch{:03d}.pth'.format(save_path, epoch))

if __name__ == '__main__':
    main()