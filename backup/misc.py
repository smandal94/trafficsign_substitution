import os, sys
import cv2
import numpy as np
import glob
from PIL import Image, ImageOps

def resize_with_padding(im, desired_size=640):
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    return Image.fromarray(new_im)

def make_gif(frame_folder, extn, cutoff = 420):
    frames = [cv2.imread(image, 1)[:,:,::-1] for image in glob.glob("{}/*{}".format(frame_folder, extn))]
    frames = [resize_with_padding(np.array(image)) for image in frames]
    frame_one = frames[0]
    frame_one.save("../examples/four_corner_err.gif", format="GIF", append_images=frames,
               save_all=True, duration=1000, loop=0)

def main():
    # files = glob.glob('/media/sayan/sayan/projects/traffic_sign/GP-GAN/results_subst/*_gpgan.png')

    # # gt = '/media/sayan/sayan/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/JPEGImages'

    # for f in files:
    #     imgname = f.split('/')[-1]
    #     gt = f.replace('_gpgan.png', '_gt.png')
    #     org = f.replace('_gpgan.png', '_img.png')

    #     gpgan = cv2.imread(f, 1)
    #     print(gpgan.shape)
    #     org = cv2.imread(org, 1)
    #     print(org.shape)
    #     gt = cv2.imread(gt, 1)
    #     print(org.shape)

    #     gpgan = cv2.resize(gpgan, (org.shape[1], org.shape[0]), None, interpolation=cv2.INTER_LANCZOS4)

    #     fn = f.replace('results_subst', 'gpgan_results')
    #     cv2.imwrite(fn, np.hstack((gt, org, gpgan)))

    #     cv2.imshow('sda', np.hstack((gt, org, gpgan)))
    #     key = cv2.waitKey(0)
    #     if key == 27:
    #         cv2.destroyAllWindows()
    #         exit()

    # files = glob.glob('/media/sayan/sayan1/projects/traffic_sign/dataset/DFG_traffic_sign_dataset/templates2/*.png')
    # for f in files:
    #     img = cv2.imread(f, -1)
    #     if img.shape[2] != 4:
    #         print(f)

    # create gif
    make_gif("../results/orb", extn='.png', cutoff=420)


if __name__ == '__main__':
    main()