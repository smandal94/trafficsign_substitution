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

def draw_label(img, box, label, color=(128, 128, 128), txt_color=(255, 255, 255), thickness=1):
    # print('box: ', box)
    lw = thickness or max(round(sum(img.shape) / 2 * 0.003), 2)  # line width
    # p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    p1, p2 = box[1], box[1]
    cv2.rectangle(img, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
                    thickness=tf, lineType=cv2.LINE_AA)
    
    return img

'''
Thanks to https://debuggercafe.com/instance-segmentation-with-pytorch-and-mask-r-cnn/
'''
def draw_segmentation_map(image, masks, boxes, labels, COLORS):
    alpha = 1 
    beta = 0.6 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    for i in range(len(masks)):
        red_map = np.zeros_like(masks[i]).astype(np.uint8)
        green_map = np.zeros_like(masks[i]).astype(np.uint8)
        blue_map = np.zeros_like(masks[i]).astype(np.uint8)
        # apply a randon color mask to each object
        # color = COLORS[random.randrange(0, len(COLORS))]
        color = COLORS[i]
        red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1]  = color
        # combine all the masks into a single image
        segmentation_map = np.stack([blue_map, green_map, red_map])
        if len(segmentation_map.shape) == 4:
            segmentation_map = segmentation_map.transpose(1, 2, 3, 0)
            segmentation_map = np.squeeze(segmentation_map, axis=0)
        else:
            segmentation_map = segmentation_map.transpose(1, 2, 0)
        # print('segmentation_map: ', segmentation_map.shape)
        # apply mask on the image
        image = cv2.addWeighted(image, alpha, segmentation_map, beta, gamma)
        # # draw the bounding boxes around the objects
        image = draw_label(image, boxes[i], labels[i])
        image = cv2.rectangle(image, boxes[i][0], boxes[i][1], color=(255, 255, 255), 
                      thickness=1)
        # # # put the label text above the objects
        # image = cv2.putText(image , labels[i], (boxes[i][0][0]-20, boxes[i][0][1]), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 
        #             thickness=1, lineType=cv2.LINE_AA)

        # cv2.imshow('image', image)
        # cv2.imshow('seg_img', segmentation_map)
        # key = cv2.waitKey(0)
        # if key == 27:
        #     cv2.destroyAllWindows()
        #     exit()
    
    return image

def main():
    imgDir = '../../dataset/DFG_traffic_sign_dataset/JPEGImages'
    test_annFile = '../../dataset/DFG_traffic_sign_dataset/DFG-tsd-annot-json/test.json'
    test_split = '../../dataset/DFG_traffic_sign_dataset/DFG-tsd-annot-json/test.txt'
    batch_size = 16
    num_workers = 12
    n_classes = 201
    threshold = 0.5
    model_path = './checkpoints/try1_epoch069.pth'
    imgsz = 800    # Trained on (800,800,3). Running inference on (1024,1024,3) for better visualization :P

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

    classnames = test_dataset.classes
    colors = test_dataset.palette
    
    
    model = get_instance_segmentation_model(n_classes).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('Model: ', model)
    print('Params: ', count_parameters(model))

    for images, targets in tqdm(test_data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            predictions = model(images)
        print('images: {}, predictions: {}'.format(len(images), len(predictions)))
        
        images = list(image.cpu() for image in images)
        targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
        predictions = [{k: v.cpu() for k, v in t.items()} for t in predictions]

        for img, target, pred in zip(images, targets, predictions):
            img = img.mul(255).permute(1, 2, 0).byte().numpy()[:,:,::-1]
            scores = list(pred['scores'].numpy())
            # index of those scores which are above a certain threshold
            thresh_preds_inidices = [scores.index(i) for i in scores if i > threshold]
            thresh_preds_count = len(thresh_preds_inidices)
            # get the masks
            masks = (pred['masks']>0.5).numpy()
            # discard masks for objects which are below threshold
            masks = masks[:thresh_preds_count]
            # get the bounding boxes, in (x1, y1), (x2, y2) format
            boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in pred['boxes']]
            # discard bounding boxes below threshold value
            boxes = boxes[:thresh_preds_count]
            # get the classes labels
            labels = [classnames[i] for i in pred['labels']]
            labels = labels[:thresh_preds_count]
            COLORS = [colors[i] for i in pred['labels']]
            COLORS = COLORS[:thresh_preds_count]

            img_id = target['image_id'][0].data

            result = draw_segmentation_map(img.copy(), masks, boxes, labels, COLORS)

            masks = target['masks'].numpy()
            boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in target['boxes']]
            labels = [classnames[i] for i in target['labels']]
            COLORS = [colors[i] for i in target['labels']]

            gt = draw_segmentation_map(img.copy(), masks, boxes, labels, COLORS)

            # result = cv2.resize(result, (0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_AREA)
            # gt = cv2.resize(gt, (0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_AREA)

            cv2.imshow('img', np.hstack([img, gt, result]))
            # cv2.imshow('img', gt)
            key = cv2.waitKey(0)
            if key == 27:
                cv2.destroyAllWindows()
                exit()
            if key == ord('s'):
                cv2.imwrite('./results/result_id{}.png'.format(img_id), np.hstack([gt, result]))
    

if __name__ == '__main__':
    main()