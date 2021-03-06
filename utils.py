#-*- coding:utf-8 -*-
#Author:Tunan
#Creation Data:2020/10/30

import os
import sys
import random
from typing import Tuple
import numpy as np
from PIL import Image

class ImagePool(object):
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.imgs = []

    def query(self, img):
        if self.pool_size == 0:
            return img

        if len(self.imgs) < self.pool_size:
            self.imgs.append(img)
            return img
        else:
            if random.random() > 0.5:
                random_id = random.randrange(0, self.pool_size)
                tmp_img = self.imgs[random_id].copy()
                self.imgs[random_id] = img.copy()
                return tmp_img
            else:
                return img


def imresize(input_image: np.ndarray, size:Tuple[int, int]) -> np.ndarray:
    """
    Reshape image to given dimensions using Pillow's resize function.
    """
    return np.array(Image.fromarray(input_image.astype('uint8')).resize(size))

def center_crop(img, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h

    h, w = img.shape[:2]
    h_start = int(round((h - crop_h) / 2.))
    w_start = int(round((w - crop_w) / 2.))
    img_crop = imresize(img[h_start: h_start + crop_h, w_start: w_start +crop_w], [resize_h, resize_w])
    return img_crop

def imread(path, is_gray_scale=False, img_size=None):
    img = np.asarray(Image.open(path)).astype(np.float)

    if not is_gray_scale and not (img.ndim == 3 and img.shape[2] == 3):
        img = np.dstack((img, img, img))

    if img_size is not None:
        img = imresize(img, img_size)

    return img

def load_data(image_path, input_height, input_width, resize_height=64, resize_width=64, crop=True, is_gray_scale=False):
    img = imread(path=image_path, is_gray_scale=is_gray_scale)

    if crop:
        cropped_img = center_crop(img, input_height, input_width, resize_height, resize_width)
    else:
        cropped_img = imresize(img, [resize_height, resize_width])

    img_trans = transform(cropped_img)

    if is_gray_scale and (img_trans.ndim == 2):
        img_trans = np.expand_dims(img_trans, axis=2)

    return img_trans

def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(extension)]

    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname) for fname in os.listdir(path) if fname.endswith(extension)]

    if sort:
        filenames = sorted(filenames)

    return filenames

def imagefiles2arrs(filenames): #zuo yong???
    img_shape = image_shape(filenames[0])
    images_arr = None

    if len(img_shape) == 3:
        images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1], img_shape[2]), dtype=np.float32)
    elif len(img_shape) == 2:
        images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1]), dtype=np.float32)

    for file_index in range(len(filenames)):
        img = Image.open(filenames[file_index])
        images_arr[file_index] = np.asarray(img).astype(np.float32)

    return images_arr

def image_shape(filename):
    img = Image.open(filename, mode="r")
    img_arr = np.asarray(img)
    img_shape = img_arr.shape
    return img_shape

def print_metrics(itr, kargs):
    print("*** Iteration {} ===>".format(itr))
    for name, value in kargs.items():
        print("{} : {},".format(name, value))
    print("")
    sys.stdout.flush()

def transform(img):
    return img / 127.5 - 1.0

def inverse_transform(img):
    return (img + 1.) / 2.