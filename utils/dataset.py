from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import SimpleITK as sitk

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader, random_split
from scipy import ndimage
from torchvision import transforms
import random

# class RandomCrop3D():
#     def __init__(self, img_sz, crop_sz):
#         c, h, w, d = img_sz
#         assert (h, w, d) > crop_sz
#         self.img_sz  = tuple((h, w, d))
#         self.crop_sz = tuple(crop_sz)
        
#     def __call__(self, x):
#         slice_hwd = [self._get_slice(i, k) for i, k in zip(self.img_sz, self.crop_sz)]
#         return self._crop(x, *slice_hwd)
        
#     @staticmethod
#     def _get_slice(sz, crop_sz):
#         try : 
#             lower_bound = torch.randint(sz-crop_sz, (1,)).item()
#             return lower_bound, lower_bound + crop_sz
#         except: 
#             return (None, None)
    
#     @staticmethod
#     def _crop(x, slice_h, slice_w, slice_d):
#         return x[:, slice_h[0]:slice_h[1], slice_w[0]:slice_w[1], slice_d[0]:slice_d[1]]



class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, img_suffix='_T2w', mask_suffix='_dseg'):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.ids = [file.replace('_T2w', '').split('.')[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    @staticmethod
    def randomCrop(img, mask, width, height, depth):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == mask.shape[0]
        assert img.shape[1] == mask.shape[1]

        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        z = random.randint(0, img.shape[2] - depth)
        img = img[y:y+height, x:x+width, z:z+depth]
        mask = mask[y:y+height, x:x+width, z:z+depth]
        return img, mask

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + self.img_suffix + '.*')
        # print(self.masks_dir + idx + self.mask_suffix + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'

        mask = sitk.ReadImage(mask_file[0])
        img = sitk.ReadImage(img_file[0])

        mask = sitk.GetArrayFromImage(mask)
        img = sitk.GetArrayFromImage(img)

        img, mask = self.randomCrop(img, mask, 32, 32, 32)
        
        img = img[np.newaxis, :]
        mask = mask[np.newaxis, :]

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'id': idx
        }

class FeTADataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir):
        super().__init__(imgs_dir, masks_dir)


if __name__ == "__main__":

    dir_img = '../data/imgs_crop/'
    dir_mask = '../data/masks_crop/'
    dir_checkpoint = 'checkpoints/'
    val_percent = 0.2

    dataset = BasicDataset(dir_img, dir_mask)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    for batch in train_loader:
        imgs = batch['image']
        true_masks = batch['mask']
        print(imgs.shape, true_masks.shape)
        # # print(true_masks.max())

        # if true_masks.max() != 7:
        #     print(batch['id'])  