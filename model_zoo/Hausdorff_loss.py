import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve

"""
Hausdorff loss implementation based on paper:
https://arxiv.org/pdf/1904.10030.pdf
"""

class HausdorffDTLoss(nn.Module):
    '''
    Binary Hausdorff loss based on distance transform
    '''

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss,self).__init__()
        self.alpha = alpha
    
    @torch.no_grad()
    def distance_field(self, img):
        field = np.zeros_like(img)
        
        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field


    def forward(self, pred, target, debug=False) :
        assert pred.dim() == 4 or pred.dim() == 5           
        ''' Only 2D and 3D supported '''
        assert (pred.dim() == target.dim())                
        ''' Prediction and target need to be of same dimension ''' 