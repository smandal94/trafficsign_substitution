import os, sys
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
import albumentations as A
from copy import deepcopy
import uuid

from alignment_correction import align

import torch
from torch.optim.lr_scheduler import StepLR
import src.utils as utils
from src.engine import train_one_epoch, evaluate

from src.dataset import DFGTrafficSignDataset
from src.model import get_instance_segmentation_model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

sign_cats  = {
            'triangles' : ['I-1', 'I-1.1', 'I-2', 'I-2.1', 'I-3', 'I-4', 'I-5', 'I-5.1', 'I-5.2', 'I-8'
                    'I-9', 'I-10', 'I-11', 'I-13', 'I-13.1', 'I-14', 'I-15', 'I-16', 'I-17', 'I-18',
                    'I-19', 'I-20', 'I-25', 'I-27', 'I-28', 'I-28.1', 'I-29', 'I-29.1', 'I-30',
                    'I-32', 'I-34', 'I-36', 'I-37', 'X-4'],
            'circles' : ['II-3', 'II-4', 'II-6', 'II-7', 'II-7.1', 'II-8', 'II-10.1', 'II-14', 'II-17',
                    'II-18', 'II-26', 'II-26.1', 'II-28', 'II-30-10', 'II-30-30', 'II-30-40',
                    'II-30-50', 'II-30-60', 'II-30-70', 'II-32', 'II-33', 'II-34', 'II-35', 'II-39',
                    'II-40', 'II-45', 'II-45.1', 'II-45.2', 'II-46', 'II-46.1', 'II-46.2', 'II-47',
                    'II-47.1', 'II-48', 'III-16', 'III-18-40', 'III-18-50', 'III-18-60', 'III-18-70',
                    'III-21', 'III-23', 'X-1.2', 'II-2'],
            'squares' : ['III-1', 'III-2', 'III-5', 'III-6', 'III-8-1', 'III-29-30', 'III-29-40', 'III-30-30',
                    'III-35', 'III-85-3', 'III-107.1-1', 'III-107.1-2', 'III-107-1', 'III-107-2',
                    'III-124', 'III-202-5', 'IV-13.1-2', 'IV-13.1-3', 'IV-13.1-4', 'IV-13-1', 'IV-13-2',
                    'IV-13-3', 'IV-13-4', 'IV-13-5', 'IV-13-6', 'IV-18', 'I-38', 'III-3', 'II-1'],
            'vertical_rect' : ['I-39-1', 'I-39-2', 'I-39-3', 'III-10', 'III-12', 'III-37', 'III-39', 'III-40',
                    'III-42', 'III-43', 'III-45', 'III-77', 'III-112', 'III-113', 'III-123',
                    'III-206-1', 'VI-3.1-1', 'VI-3.1-2', 'VI-3-1', 'VI-3-2', 'VI-8'],
            'horizontal_rect' : ['III-33', 'III-34', 'III-85-2', 'III-107.2-1', 'III-107.2-2', 'III-120',
                    'III-120.1', 'IV-12', 'IV-12.1', 'IV-16', 'IV-17', 'IV-20-1', 'VI-2.1'],
            'arbitrary' : ['II-19-4', 'II-21', 'II-22', 'II-23', 'II-41', 'II-42', 'II-42.1', 'II-43', 'III-14',
                    'III-14.1', 'III-15', 'III-25', 'III-25.1', 'III-27', 'III-46', 'III-47', 'III-50',
                    'III-54', 'III-59', 'III-64', 'III-68', 'III-74', 'III-78', 'III-84', 'III-84-1',
                    'III-85.1', 'III-86-1', 'III-86-2', 'III-87', 'III-90', 'III-90.1', 'III-90.2',
                    'III-91', 'III-105', 'III-105.1', 'III-105.3', 'III-120-1', 'III-203-2', 'IV-1',
                    'IV-1.1', 'IV-2', 'IV-3-1', 'IV-3-2', 'IV-3-4', 'IV-3-5', 'IV-5', 'IV-6', 'IV-10',
                    'IV-11', 'VII-4', 'VII-4.1-1', 'VII-4.3', 'VII-4.3-1', 'VII-4.3-2', 'VII-4.4-1',
                    'VII-4.4-2', 'VII-4-1', 'VII-4-2', 'X-1.1', 'X-6-3']
        }

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

def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return True 
    else:
        return False

def main():
    imgDir = '../../dataset/DFG_traffic_sign_dataset/JPEGImages'
    test_annFile = '../../dataset/DFG_traffic_sign_dataset/DFG-tsd-annot-json/test.json'
    test_split = '../../dataset/DFG_traffic_sign_dataset/DFG-tsd-annot-json/test.txt'
    batch_size = 16
    num_workers = 12
    n_classes = 201
    threshold = 0.5
    model_path = './checkpoints/maskrcnn_epoch069.pth'
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

    template_dir = '/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/templates'

    # target_cls = ['I-1.1', 'I-1', 'I-2.1', 'I-2', 'I-3', 'I-4', 'I-5.1', 'I-5.2', 'I-5', 'I-8', 'I-9', 'I-10'
    #             'I-11', 'I-13.1', 'I-13', 'I-14', 'I-15', 'I-16', 'I-17', 'I-18', 'I-19', 'I-20', 'I-25'
    #             'I-27', 'I-28.1', 'I-28', 'I-29.1', 'I-29', 'I-30', 'I-32', 'I-34', 'I-36', 'I-37']


    lama_dir = '/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/results/DFG_lama/test/output'
    save_dir = '/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/results/DFG_resultcrops/iter2'
    # img = cv2.imread('results_subst/0000026_cp.png', 1)
    # cv2.imshow('final', img)
    # key = cv2.waitKey(0)
    # if key == 27:
    #     cv2.destroyAllWindows()
    #     exit()

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
            

            # if not common_member(labels, target_cls):
            #     continue
            lama_path = '{}/{}_mask.png'.format(lama_dir, img_path.split('/')[-1][:-4])
            lama_img = cv2.imread(lama_path, 1)
            assert img.shape == lama_img.shape


            gt = draw_segmentation_map(img.copy(), masks, boxes, labels, COLORS)

            poisson = deepcopy(lama_img)
            copy_paste = deepcopy(lama_img)
            new_mask = np.zeros((img.shape[1], img.shape[0]), dtype=np.uint8)
            src = np.zeros_like(img, dtype=np.uint8)

            for msk, bxs, lbl, annot_id in zip(masks, boxes, labels, annot_ids):
                # if lbl not in target_cls:
                #     continue

                check_cat = None
                for category in sign_cats.keys():
                    if lbl in sign_cats[category]:
                        check_cat = category
                        break
                if check_cat == 'arbitrary':
                    continue
                
                new = '{}/{}.png'.format(template_dir, lbl)
                new = cv2.imread(new, -1)

                pt1, pt2 = bxs
                xmin, ymin = pt1
                xmax, ymax = pt2
                
                msk *= 255

                crop = img[ymin:ymax, xmin:xmax, :]
                alpha1 = msk[ymin:ymax, xmin:xmax]
                alpha = cv2.cvtColor(alpha1, cv2.COLOR_GRAY2RGB)

                bg = np.zeros_like(crop, np.uint8)

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

                # template_path = '/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/templates/{}.png'.format(lbl)
                # template = cv2.imread(template_path, -1)
                template = deepcopy(new)

                aligned_template, aligned_result = align(template, final_crop, new)

                
                new_mask[ymin:ymax, xmin:xmax] = aligned_result[:,:,-1]

                # copy-paste
                crop = lama_img[ymin:ymax, xmin:xmax, :]
                alpha1 = aligned_result[:,:,-1]
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
                center = ((xmin+xmax)//2, (ymin+ymax)//2)
                try:
                    poisson = cv2.seamlessClone(aligned_result[:,:,:-1], poisson, alpha1, center, cv2.NORMAL_CLONE)
                except:
                    poisson[ymin:ymax, xmin:xmax] = final_crop2[:,:,:]
                # cv2.imshow('FINAL', final_crop2)
                # cv2.imshow('mask', alpha1)
                # key = cv2.waitKey(0)
                # if key == 27:
                #     cv2.destroyAllWindows()
                #     exit()

                save_pathR = '{}/real/{:06d}.png'.format(save_dir, annot_id)
                save_pathS1 = '{}/synthetic_poisson/{:06d}.png'.format(save_dir, annot_id)
                save_pathS2 = '{}/synthetic_cp/{:06d}.png'.format(save_dir, annot_id)
                # cv2.imwrite(save_pathS1, poisson[ymin:ymax, xmin:xmax, :])
                # cv2.imwrite(save_pathS2, copy_paste[ymin:ymax, xmin:xmax, :])
                # cv2.imwrite(save_pathR, img[ymin:ymax, xmin:xmax, :])

                # cv2.imshow('img', np.hstack((final_crop, aligned_template, aligned_result)))
                # cv2.imshow('new_mask', new_mask)
                # cv2.imshow('src', src)
                # cv2.imshow('final', aligned_result)
                # key = cv2.waitKey(0)
                # if key == 27:
                #     cv2.destroyAllWindows()
                #     exit()

            _, new_mask = cv2.threshold(new_mask,127,255,cv2.THRESH_BINARY)
            new_mask = new_mask.astype(np.uint8)
            new_mask = new_mask/255

            all = np.hstack((gt, img, lama_img, copy_paste))
            cv2.imshow('img', all)
            # cv2.imshow('new_mask', new_mask)
            gt_file = './results_subst/{:07d}_gt.png'.format(img_id)
            img_file = './results_subst/{:07d}_img.png'.format(img_id)
            cp_file = './results_subst/{:07d}_cp.png'.format(img_id)
            src_file = './results_subst/{:07d}_src.png'.format(img_id)
            mask_file = './results_subst/{:07d}_mask.png'.format(img_id)
            all_file = './results_subst/{:07d}_gt_img_poisson.png'.format(img_id)
            key = cv2.waitKey(0)
            if key == 27:
                cv2.destroyAllWindows()
                exit()
            # elif key == ord('s'):
            #     # gt = gt[420:1920-420,:,:]
            #     # cv2.imwrite(gt_file, gt)
            #     # img = img[420:1920-420,:,:]
            #     # cv2.imwrite(img_file, img)
            #     # poisson = poisson[420:1920-420,:,:]
            #     # cv2.imwrite(cp_file, poisson)
            #     # src = src[420:1920-420,:,:]
            #     # cv2.imwrite(src_file, src)
            #     # new_mask = new_mask[420:1920-420,:]
            #     # cv2.imwrite(mask_file, new_mask)
            #     all = all[420:1920-420,:]
            #     # cv2.imwrite(all_file, all)



if __name__ == '__main__':
    main()