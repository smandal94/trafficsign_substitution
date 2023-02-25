import os, sys
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
import albumentations as A

import torch
from torch.optim.lr_scheduler import StepLR
import src.utils as utils
from src.engine import train_one_epoch, evaluate

from src.dataset import DFGTrafficSignDataset
from src.model import get_instance_segmentation_model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    imgDir = '../../dataset/DFG_traffic_sign_dataset/JPEGImages'
    test_annFile = '../../dataset/DFG_traffic_sign_dataset/DFG-tsd-annot-json/test.json'
    test_split = '../../dataset/DFG_traffic_sign_dataset/DFG-tsd-annot-json/test.txt'
    batch_size = 16
    num_workers = 12
    n_classes = 201
    model_path = './checkpoints/try1_epoch069.pth'
    imgsz = 1024

    test_transform = A.Compose(
        [
            A.LongestMaxSize(imgsz, interpolation=1),
            A.PadIfNeeded(min_height=imgsz, min_width=imgsz, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1)
        ],
        bbox_params={'format':'pascal_voc', 'min_area': 0, 'min_visibility': 0, 'label_fields': ['category_id']}, p=1
    )
    test_dataset = DFGTrafficSignDataset(imgDir, test_split, test_annFile, test_transform)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers//2,
                                            collate_fn=utils.collate_fn)
    
    
    model = get_instance_segmentation_model(n_classes).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('Model: ', model)
    print('Params: ', count_parameters(model))

    evaluate(model, test_data_loader, device=device)

if __name__ == '__main__':
    main()
