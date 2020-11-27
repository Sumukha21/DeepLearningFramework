import os
import cv2
import numpy as np


def image_reader(image_path, image_folder=None, read_mode=-1, flip=False):
    try:
        if image_folder is None:
            img = cv2.imread(image_path, read_mode)
        else:
            img = cv2.imread(os.path.join(image_folder, image_path), read_mode)
        if flip:
            img = np.flip(img, 2)
        return img
    except FileNotFoundError as e:
        print("Provided image path %s does not exist" % image_path)


def image_resize(img, target_size, resize_mode=1):
    resize_modes = [cv2.INTER_NEAREST, cv2.INTER_LINEAR]
    try:
        if resize_mode not in [0, 1]:
            raise AttributeError("Resize modes should be either 0 (Nearest interpolation) or 1 (Linear interpolation)")
        if len(img.shape) == 3 and img.shape[-1] == 1:
            img = cv2.resize(img, tuple(target_size), resize_modes[resize_mode])
            img = img[:, :, np.newaxis]
        else:
            img = cv2.resize(img, tuple(target_size), resize_modes[resize_mode])
        return img
    except AttributeError as e:
        print(e)


def image_normalize(img, mean_std=None):
    if mean_std is not None:
        img = (img - mean_std[0]) / mean_std[1]
    else:
        img = img / 255
    return img


def image_writer(img, save_path, save_folder=None):
    if save_folder is not None:
        save_path = os.path.join(save_folder, save_path)
    cv2.imwrite(save_path, img)
