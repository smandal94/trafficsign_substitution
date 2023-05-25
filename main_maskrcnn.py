import os, sys
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
import albumentations as A
from copy import deepcopy
import uuid
import random
import os
import torch
from pycocotools.coco import COCO
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import json
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon

from match_pairs import align_superglue

from models.networks import RainNet
from models.normalize import RAIN
from util.config import cfg as test_cfg
from util import util
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

new_classes = ['__background__', 'I-1', 'I-1.1', 'I-10', 'I-11', 'I-13', 'I-13.1', 'I-14', 'I-15', 'I-16', 'I-17', 'I-18', 'I-19', 
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

new_dict = {
    'images': [],
    'categories': [],
    'annotations': [],
    'info': []
}

# images: [['id', 'height', 'width', 'file_name']]
# categories: [['id', 'name', 'supercategory']] 
# annotations: [['id', 'area', 'bbox', 'category_id', 'segmentation', 'image_id', 'ignore', 'iscrowd']]
# info: ['contributor', 'description', 'date_created', 'year', 'version']

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
    # net.load_state_dict(state_dict)
    return net


def main():
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

    template_dir = '/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/templates2'
    lama_dir = '/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/DFG_lama/test/output'
    save_dir = '/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/results/DFG_resultcrops/iter5_maskrcnn'
    img_dir = '/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/JPEGImages'
    annFile = '/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/DFG-tsd-annot-json/arbitrary_removed/test.json'
    with open(annFile, 'r') as f:
        train_file = json.load(f)
    print(len(train_file['categories']))

    # info
    new_dict['info'] = train_file['info']

    # categories
    new_dict['categories'] = train_file['categories']
    # print(new_dict)
    
    # annotations and images
    coco = COCO(annFile)
    imgIDs = coco.getImgIds()

    for idx in tqdm(range(len(imgIDs))):
        # img_data = coco.loadImgs(923)[0]
        img_data = coco.loadImgs(imgIDs[idx])[0]
        img_path = '{}/{}'.format(img_dir, img_data['file_name'])
        annIds = coco.getAnnIds(imgIds=img_data['id'], iscrowd=None)
        annots = coco.loadAnns(annIds)

        # orginal img
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)[:,:,::-1]
        # LAMA inpaint foreground removed image
        lama_path = '{}/{}_mask.png'.format(lama_dir, img_path.split('/')[-1][:-4])
        lama_img = cv2.imread(lama_path, 1)[175:-175,:,:]
        lama_img = cv2.resize(lama_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        copy_paste = deepcopy(lama_img)
        copy_paste_org = deepcopy(lama_img)
        new_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        matched_results = []
        done = False

        for i in range(len(annots)):
            fg_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

            area = annots[i]["area"]
            annot_id = annots[i]["id"]
            lbl = new_classes[annots[i]['category_id']]
            cat_id = new_classes.index(lbl)
            mask = coco.annToMask(annots[i])
            bbox = np.array(annots[i]['bbox'], dtype=np.int32)
            image_id = annots[i]['image_id']
            ignore = annots[i]["ignore"]
            iscrowd = annots[i]['iscrowd']
            xmin = max(bbox[0], 0)
            ymin = max(bbox[1], 0)
            xmax = min(bbox[0]+bbox[2], img_data['width'])
            ymax = min(bbox[1]+bbox[3], img_data['height'])

            for category in sign_cats.keys():
                if lbl in sign_cats[category]:
                    check_cat = category

            choices = deepcopy(sign_cats[check_cat])
            choices.remove(lbl)
            choice = random.choice(choices)
            choice_id = new_classes.index(choice)

            new = '{}/{}.png'.format(template_dir, choice)
            new = cv2.imread(new, -1)

            # white BGRA substitution
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

            mask *= 255

            crop = img[ymin:ymax, xmin:xmax, :]
            alpha1 = mask[ymin:ymax, xmin:xmax]
            alpha = cv2.cvtColor(alpha1, cv2.COLOR_GRAY2RGB)

            bg = np.ones_like(crop, np.uint8) * 255

            # white BGRA foreground 
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

            # white BGRA template
            template_path = '{}/{}.png'.format(template_dir, lbl)
            # print(lbl, template_path)
            template = cv2.imread(template_path, -1)
            b, g, r, a1 = cv2.split(template)
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
            template = cv2.merge((b, g, r, a1))

            # homography based projective transformation
            aligned_template, aligned_result, out, out2 = align_superglue(template, final_crop, new, check_cat)
            matched_results.append([out, out2])

            new_mask[ymin:ymax, xmin:xmax] = aligned_result[:,:,-1]
            fg_mask[ymin:ymax, xmin:xmax] = aligned_result[:,:,-1]
            contours = measure.find_contours(fg_mask, 0.5, positive_orientation='low')
            segmentations = []
            polygons = []
            for contour in contours:
                # Flip from (row, col) representation to (x, y)
                # and subtract the padding pixel
                for j in range(len(contour)):
                    row, col = contour[j]
                    contour[j] = (col - 1, row - 1)

                # Make a polygon and simplify it
                poly = Polygon(contour)
                poly = poly.simplify(1.0, preserve_topology=False)
                if(poly.is_empty):
                    continue
                polygons.append(poly)
                segmentation = np.array(poly.exterior.coords).ravel().tolist()
                segmentations.append(segmentation)
            # Combine the polygons to calculate the bounding box and area
            multi_poly = MultiPolygon(polygons)
            x, y, max_x, max_y = multi_poly.bounds
            width = max_x - x
            height = max_y - y
            bbox = (x, y, width, height)
            area = multi_poly.area
            annotation = {
                            'segmentation': segmentations,
                            'iscrowd': iscrowd,
                            'image_id': image_id,
                            'category_id': choice_id,
                            'id': annot_id,
                            'bbox': bbox,
                            'area': area
                        }
            if area > 100:
                new_dict['annotations'].append(annotation)
            else:
                print("<100: ", annotation)
                continue

            try:
                cocomask = coco.annToMask(annotation)
            except:
                print("annToMask fail: ", annotation)
                continue
            
            # copy-paste new
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

            copy_paste[ymin:ymax, xmin:xmax] = final_crop2[:,:,:]

            # copy-paste org
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
            done = True

            # print(lbl, choice, annots[i],annots[i]['category_id'], new_classes.index('II-47'))
            # exit()

            # cv2.imshow('img1', final_crop)
            # fg_mask = cv2.resize(fg_mask, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            # cv2.imshow('img2', fg_mask)
            # key = cv2.waitKey(0)
            # if key == 27:
            #     print(lbl, choice, annots[i], new_classes[annots[i]['category_id']], new_classes.index('II-47'))
            #     cv2.destroyAllWindows()
            #     exit()
        
        # done = False
        if done:
            new_dict['images'].append(img_data)
            _, new_mask = cv2.threshold(new_mask,127,255,cv2.THRESH_BINARY)
            new_mask = new_mask.astype(np.uint8)

            comp = rainnet_transform_image(Image.fromarray(copy_paste[:,:,::-1])).unsqueeze(0).to(device)
            mask = rainnet_transform_mask(Image.fromarray(new_mask[:,:])).unsqueeze(0).to(device)
            real = rainnet_transform_image(Image.fromarray(lama_img[:,:,::-1])).unsqueeze(0).to(device)
            pred = rainnet.processImage(comp, mask, real)
            pred_rgb = util.tensor2im(pred[0:1])[:,:,::-1]
            pred_final = cv2.resize(pred_rgb, (img.shape[1], img.shape[0]), None, interpolation=cv2.INTER_LANCZOS4)

            # 1 LAMA
            lama_file = '{}/lama/{}_gt_lama.png'.format(save_dir, img_path.split('/')[-1][:-4])
            l_img = np.hstack((img, lama_img))
            cv2.imwrite(lama_file, l_img)

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
            cv2.imwrite(subst_file, sub_img)

            # 4 RainNet
            rain_file = '{}/rainnet/{}_gt_rain.png'.format(save_dir, img_path.split('/')[-1][:-4])
            rain_img = np.hstack((copy_paste, pred_final))
            cv2.imwrite(rain_file, rain_img)

            # 5 template
            temp_file = '{}/template/{}_gt_template.png'.format(save_dir, img_path.split('/')[-1][:-4])
            temp_img = np.hstack((img, copy_paste_org))
            cv2.imwrite(temp_file, temp_img)

            # 6 results
            subst_file = '{}/results/{}'.format(save_dir, img_data['file_name'])
            cv2.imwrite(subst_file, pred_final)

            # cv2.imshow('img2', pred_final)
            # cv2.imshow('sda', new_mask)
            # key = cv2.waitKey(0)
            # if key == 27:
            #     print(lbl, choice)
            #     cv2.destroyAllWindows()
            #     exit()

    with open("/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/DFG-tsd-annot-json/arbitrary_removed/test_substitution.json", "w") as outfile:
        json.dump(new_dict, outfile)


if __name__ == '__main__':
    main()