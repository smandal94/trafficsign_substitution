import os, sys
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
import albumentations as A
from copy import deepcopy
import uuid
import random

from match_pairs import align_superglue

import torch
from torch.optim.lr_scheduler import StepLR
import src.utils as utils
from src.engine import train_one_epoch, evaluate

from src.dataset import DFGTrafficSignDataset
from src.model import get_instance_segmentation_model

from models.networks import RainNet
from models.normalize import RAIN
from util.config import cfg as test_cfg
from util import util
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

sign_cats  = {
            'triangles' : ['I-1', 'I-1.1', 'I-2', 'I-2.1', 'I-3', 'I-4', 'I-5', 'I-5.1', 'I-5.2', 'I-8',
                    'I-9', 'I-10', 'I-11', 'I-13', 'I-13.1', 'I-14', 'I-15', 'I-16', 'I-17', 'I-18',
                    'I-19', 'I-20', 'I-25', 'I-27', 'I-28', 'I-28.1', 'I-29', 'I-29.1', 'I-30',
                    'I-32', 'I-34', 'I-36', 'I-37', 'X-4'],
            'circles' : ['II-3', 'II-4', 'II-6', 'II-7', 'II-7.1', 'II-8', 'II-10.1', 'II-14', 'II-17',
                    'II-18', 'II-26', 'II-26.1', 'II-28', 'II-30-10', 'II-30-30', 'II-30-40',
                    'II-30-50', 'II-30-60', 'II-30-70', 'II-32', 'II-35', 'II-39',
                    'II-40', 'II-45', 'II-45.1', 'II-45.2', 'II-46', 'II-46.1', 'II-46.2', 'II-47',
                    'II-47.1', 'III-16', 'III-18-40', 'III-18-50', 'III-18-60', 'III-18-70',
                    'III-21', 'III-23', 'II-2'],
            'squares' : ['III-1', 'III-2', 'III-5', 'III-6', 'III-8-1', 'III-29-30', 'III-29-40', 'III-30-30',
                    'III-35', 'III-107.1-1', 'III-107.1-2', 'III-107-1', 'III-107-2',
                    'III-124', 'III-202-5', 'IV-13.1-2',
                    'IV-13-5', 'IV-13-6', 'IV-18', 'I-38', 'II-1', 'III-3'],
            'vertical_rect' : ['I-39-1', 'I-39-2', 'I-39-3', 'III-10', 'III-12',
                    'III-112', 'III-123',
                    'VI-3.1-1', 'VI-3.1-2', 'VI-3-1', 'VI-3-2'],
            'horizontal_rect' : ['III-33', 'III-34', 'III-107.2-1', 'III-107.2-2',
                    'III-120.1', 'IV-12', 'IV-12.1', 'IV-17', 'IV-20-1', 'VI-2.1'],
            'arbitrary' : ['II-19-4', 'II-21', 'II-22', 'II-23', 'II-41', 'II-42', 'II-42.1', 'II-43', 'III-14',
                    'III-14.1', 'III-15', 'III-25', 'III-25.1', 'III-27', 'III-46', 'III-47', 'III-50',
                    'III-54', 'III-59', 'III-64', 'III-68', 'III-74', 'III-78', 'III-84', 'III-84-1',
                    'III-85.1', 'III-86-1', 'III-86-2', 'III-87', 'III-90', 'III-90.1', 'III-90.2',
                    'III-91', 'III-105', 'III-105.1', 'III-105.3', 'III-120-1', 'III-203-2', 'IV-1',
                    'IV-1.1', 'IV-2', 'IV-3-1', 'IV-3-2', 'IV-3-4', 'IV-3-5', 'IV-5', 'IV-6', 'IV-10',
                    'IV-11', 'VII-4', 'VII-4.1-1', 'VII-4.3', 'VII-4.3-1', 'VII-4.3-2', 'VII-4.4-1',
                    'VII-4.4-2', 'VII-4-1', 'VII-4-2', 'X-1.1', 'X-6-3', 'VI-8', 'II-48', 'II-33', 'III-37',
                    'III-39', 'III-40', 'III-42', 'III-43', 'III-45', 'III-77', 'III-85-2', 'III-85-3',
                    'III-113', 'III-120','III-206-1','IV-13-4', 'IV-13-3','IV-13-2','IV-13-1', 'IV-13.1-4',
                    'IV-13.1-3', 'IV-16','X-1.2', 'II-34']
        }

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# RainNet
def load_network(cfg):
    net = RainNet(input_nc=cfg.input_nc, 
                output_nc=cfg.output_nc, 
                ngf=cfg.ngf, 
                norm_layer=RAIN, 
                use_dropout=not cfg.no_dropout)
    
    load_path = os.path.join(cfg.checkpoints_dir, cfg.name, 'netG_latest.pth')
    if not os.path.exists(load_path):
        print('load_path: ', load_path)
        print('%s not exists. Please check the file'%(load_path))
        raise FileExistsError
    print(f'loading the model from {load_path}')
    state_dict = torch.load(load_path)
    util.copy_state_dict(net.state_dict(), state_dict)
    return net

def draw_label(img, box, label, color=(0, 0, 0), txt_color=(255, 255, 255), thickness=3):
    # print('box: ', box)
    lw = thickness or max(round(sum(img.shape) / 2 * 0.003), 2)  # line width
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
    beta = 1.0 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    for i in range(len(masks)):
        red_map = np.zeros_like(masks[i]).astype(np.uint8)
        green_map = np.zeros_like(masks[i]).astype(np.uint8)
        blue_map = np.zeros_like(masks[i]).astype(np.uint8)
        color = COLORS[i]
        red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1]  = color
        # combine all the masks into a single image
        segmentation_map = np.stack([blue_map, green_map, red_map])
        if len(segmentation_map.shape) == 4:
            segmentation_map = segmentation_map.transpose(1, 2, 3, 0)
            segmentation_map = np.squeeze(segmentation_map, axis=0)
        else:
            segmentation_map = segmentation_map.transpose(1, 2, 0)
        # apply mask on the image
        image = cv2.addWeighted(image, alpha, segmentation_map, beta, gamma)
        # # draw the bounding boxes around the objects
        image = draw_label(image, boxes[i], labels[i])
        image = cv2.rectangle(image, boxes[i][0], boxes[i][1], color=(255, 0, 255), 
                      thickness=2)
        # # # put the label text above the objects
        # image = cv2.putText(image , labels[i], (boxes[i][0][0]-20, boxes[i][0][1]), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 
        #             thickness=2, lineType=cv2.LINE_AA)
        # cv2.imshow('image', image)
        # cv2.imshow('seg_img', segmentation_map)
        # key = cv2.waitKey(0)
        # if key == 27:
        #     cv2.destroyAllWindows()
        #     exit()
    
    return image

def main():
    imgDir = './examples/DFG'
    test_annFile = './examples/test.json'
    test_split = './examples/demo.txt'
    batch_size = 16
    num_workers = 12
    imgsz = 1920  # 800
    cutoff = 420 # 175

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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rainnet = load_network(test_cfg)
    rainnet = rainnet.to(device)

    rainnet_sz = 1536 # 768

    rainnet_transform_image = transforms.Compose([
        transforms.Resize([rainnet_sz, rainnet_sz]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    rainnet_transform_mask = transforms.Compose([
        transforms.Resize([rainnet_sz, rainnet_sz]),
        transforms.ToTensor()
    ])

    template_dir = './examples/templates'
    gt_dir = './examples/DFG_labels'
    lama_dir = './examples/inpainted'
    save_dir = './results'
    

    for images, targets, img_paths in tqdm(test_data_loader):
        images = list(image.cpu() for image in images)
        targets = [{k: v.cpu() for k, v in t.items()} for t in targets]

        for img, target, img_path in zip(images, targets, img_paths):
            img = img.mul(255).permute(1, 2, 0).byte().numpy()[:,:,::-1]

            masks = target['masks'].numpy()
            boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in target['boxes']]
            labels = [classnames[i] for i in target['labels']]
            COLORS = [colors[i] for i in target['labels']]
            img_id = target["image_id"].numpy()[0]
            annot_ids = target["annot_ids"].numpy()
            
            lama_path = '{}/{}_mask.png'.format(lama_dir, img_path.split('/')[-1][:-4])
            lama_img = cv2.imread(lama_path, 1)
            lama_img = cv2.resize(lama_img, (imgsz, imgsz), interpolation=cv2.INTER_LANCZOS4)

            gt = draw_segmentation_map(img.copy(), masks, boxes, labels, COLORS)
            gt_path = '{}/{}_annots.png'.format(gt_dir, img_path.split('/')[-1][:-4])
            cv2.imwrite(gt_path, gt[cutoff:-cutoff,:,:])

            poisson = deepcopy(lama_img)
            copy_paste = deepcopy(lama_img)
            copy_paste_org = deepcopy(lama_img)
            new_mask = np.zeros((img.shape[1], img.shape[0]), dtype=np.uint8)
            src = np.zeros_like(img, dtype=np.uint8)

            matched_results = []
            done = False

            for msk, bxs, lbl, annot_id in zip(masks, boxes, labels, annot_ids):
                check_cat = None
                for category in sign_cats.keys():
                    if lbl in sign_cats[category]:
                        check_cat = category
                if check_cat == 'arbitrary':
                    continue
                
                choices = deepcopy(sign_cats[check_cat])
                choices.remove(lbl)
                choice = random.choice(choices)

                new = '{}/{}.png'.format(template_dir, choice)
                new = cv2.imread(new, -1)

                b, g, r, a1 = cv2.split(new)
                a = cv2.cvtColor(a1, cv2.COLOR_GRAY2RGB)
                temp1 = cv2.merge((b, g, r))
                bg = np.ones_like(temp1, np.uint8) * 255
                fg = temp1.astype(float) / 255.0
                bg = bg.astype(float) / 255.0
                a = a.astype(float) / 255.0
                fg = cv2.multiply(a, fg)
                bg = cv2.multiply(1.0 - a, bg)
                temp2 = cv2.add(fg, bg)
                temp2 = temp2 * 255.0
                temp2 = temp2.astype(np.uint8)
                b, g, r = cv2.split(temp2)
                new = cv2.merge((b, g, r, a1))

                pt1, pt2 = bxs
                xmin, ymin = pt1
                xmax, ymax = pt2
                
                msk *= 255

                crop = img[ymin:ymax, xmin:xmax, :]
                try:
                    alpha1 = msk[ymin:ymax, xmin:xmax]
                    alpha = cv2.cvtColor(alpha1, cv2.COLOR_GRAY2RGB)
                except:
                    print('alpha1: {}, mask: {}, {}, {}, {}, {}'.format(alpha1.shape, msk.shape, xmin, ymin, xmax, ymax))
                    continue

                # do homography with objects in black background or white background
                # bg = np.zeros_like(crop, np.uint8)
                bg = np.ones_like(crop, np.uint8) * 255
                fg = crop.astype(float) / 255.0
                bg = bg.astype(float) / 255.0
                alpha = alpha.astype(float) / 255.0
                fg = cv2.multiply(alpha, fg)
                bg = cv2.multiply(1.0 - alpha, bg)
                final_crop = cv2.add(fg, bg)
                final_crop = final_crop * 255.0
                final_crop = final_crop.astype(np.uint8)
                b, g, r = cv2.split(final_crop)
                final_crop = cv2.merge((b, g, r, alpha1))

                template_path = '{}/{}.png'.format(template_dir, lbl)
                template = cv2.imread(template_path, -1)

                b, g, r, a1 = cv2.split(template)
                a = cv2.cvtColor(a1, cv2.COLOR_GRAY2RGB)
                temp1 = cv2.merge((b, g, r))
                # do homography with objects in black background or white background
                # bg = np.zeros_like(crop, np.uint8)
                bg = np.ones_like(temp1, np.uint8) * 255
                fg = temp1.astype(float) / 255.0
                bg = bg.astype(float) / 255.0
                a = a.astype(float) / 255.0
                fg = cv2.multiply(a, fg)
                bg = cv2.multiply(1.0 - a, bg)
                temp2 = cv2.add(fg, bg)
                temp2 = temp2 * 255.0
                temp2 = temp2.astype(np.uint8)
                b, g, r = cv2.split(temp2)
                template = cv2.merge((b, g, r, a1))
                
                aligned_template, aligned_result, out, out2 = align_superglue(template, final_crop, new, check_cat)
                matched_results.append([out, out2])
                new_mask[ymin:ymax, xmin:xmax] = aligned_result[:,:,-1]

                # copy-paste new ### sign-substitution
                crop = lama_img[ymin:ymax, xmin:xmax, :]
                alpha1 = aligned_result[:,:,-1]
                alpha1 = cv2.GaussianBlur(alpha1, (3,3), 0)
                alpha = cv2.cvtColor(alpha1, cv2.COLOR_GRAY2RGB)
                fg = aligned_result[:,:,:-1].astype(float) / 255.0
                bg = crop.astype(float) / 255.0
                alpha = alpha.astype(float) / 255.0
                fg = cv2.multiply(alpha, fg)
                bg = cv2.multiply(1.0 - alpha, bg)
                final_crop2 = cv2.add(fg, bg)
                final_crop2 = final_crop2 * 255.0
                final_crop2 = final_crop2.astype(np.uint8)

                src[ymin:ymax, xmin:xmax] = aligned_result[:,:,:-1]
                copy_paste[ymin:ymax, xmin:xmax] = final_crop2[:,:,:]

                # copy-paste org ### substitute template for maybe use in training harmonization
                crop = lama_img[ymin:ymax, xmin:xmax, :]
                alpha1 = aligned_template[:,:,-1]
                alpha1 = cv2.GaussianBlur(alpha1, (3,3), 0)
                alpha = cv2.cvtColor(alpha1, cv2.COLOR_GRAY2RGB)
                fg = aligned_template[:,:,:-1].astype(float) / 255.0
                bg = crop.astype(float) / 255.0
                alpha = alpha.astype(float) / 255.0
                fg = cv2.multiply(alpha, fg)
                bg = cv2.multiply(1.0 - alpha, bg)
                final_crop2 = cv2.add(fg, bg)
                final_crop2 = final_crop2 * 255.0
                final_crop2 = final_crop2.astype(np.uint8)

                copy_paste_org[ymin:ymax, xmin:xmax] = final_crop2[:,:,:]

                # if a substitution has been made
                done = True

            # done = False
            if done:
                _, new_mask = cv2.threshold(new_mask,127,255,cv2.THRESH_BINARY)
                new_mask = new_mask.astype(np.uint8)
                
                comp = rainnet_transform_image(Image.fromarray(copy_paste[cutoff:-cutoff,:,::-1])).unsqueeze(0).to(device)
                mask = rainnet_transform_mask(Image.fromarray(new_mask[cutoff:-cutoff,:])).unsqueeze(0).to(device)
                real = rainnet_transform_image(Image.fromarray(lama_img[cutoff:-cutoff,:,::-1])).unsqueeze(0).to(device)
                pred = rainnet.processImage(comp, mask, real)

                pred_rgb = util.tensor2im(pred[0:1])[:,:,::-1]
                pred_rgb = cv2.resize(pred_rgb, (1920, 1080), None, interpolation=cv2.INTER_AREA)
                pred_final = np.zeros_like(img, dtype=np.uint8)
                pred_final[cutoff:-cutoff,:,:] = pred_rgb

                # 1 LAMA
                lama_file = '{}/lama/{}_gt_lama.png'.format(save_dir, img_path.split('/')[-1][:-4])
                l_img = np.hstack((img, lama_img))
                cv2.imwrite(lama_file, l_img[cutoff:-cutoff,:,:])

                # 2 SuperGlue
                glue_folder = '{}/superglue/{}'.format(save_dir, img_path.split('/')[-1][:-4])
                if not os.path.exists(glue_folder):
                    os.makedirs(glue_folder)
                for idm, ims in enumerate(matched_results):
                    gfile = '{}/{:04d}_match.jpg'.format(glue_folder, idm)
                    cv2.imwrite(gfile, ims[0])
                    gfile = '{}/{:04d}_trasform.jpg'.format(glue_folder, idm)
                    cv2.imwrite(gfile, ims[1])
                
                # 3 substitution
                subst_file = '{}/substitution/{}_gt_subst.png'.format(save_dir, img_path.split('/')[-1][:-4])
                sub_img = np.hstack((img, pred_final))
                cv2.imwrite(subst_file, sub_img[cutoff:-cutoff,:,:])

                # 4 RainNet
                rain_file = '{}/rainnet/{}_gt_rain.png'.format(save_dir, img_path.split('/')[-1][:-4])
                rain_img = np.hstack((copy_paste, pred_final))
                cv2.imwrite(rain_file, rain_img[cutoff:-cutoff,:,:])

                # 5 template
                temp_file = '{}/template/{}_gt_template.png'.format(save_dir, img_path.split('/')[-1][:-4])
                temp_img = np.hstack((img, copy_paste_org))
                cv2.imwrite(temp_file, temp_img[cutoff:-cutoff,:,:])

if __name__ == '__main__':
    main()