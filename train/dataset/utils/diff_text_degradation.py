#!/usr/bin/python
# encoding: utf-8
import random
import numpy as np

import torch
import torch.nn.functional as F

from .real_esrgan_bsrgan_degradation import real_esrgan_degradation, bsrgan_degradation


class Diff_Text_Degrade(object):
    def __init__(self, imgH, imgW, lq_image_probability=0.5, sr_ratio=0):
        self.imgH = imgH
        self.imgW = imgW
        self.lq_image_probability = lq_image_probability
        self.sr_ratio = sr_ratio

    def diff_multi_random_degrade(self, hq_text_image):
        degradation_type = random.random()
        if degradation_type < self.lq_image_probability: # real-esrgan
            # input should be BGR 0~1 numpy H*W*C
            # output is RGB 0~1 tensor
            if self.sr_ratio == 0:
                self.sr_ratio = random.choice([1,2,4])
            lq_text_image = real_esrgan_degradation(hq_text_image[:,:,::-1], insf=self.sr_ratio).squeeze(0).detach().numpy() # output numpy c*h*w 0~1 RGB
            lq_text_image = lq_text_image.transpose((1,2,0)) # transfer to h*w*c

        else: # bsrgan
            # input should be RGB 0~1 numpy H*W*C
            # output is RGB 0~1 numpy H*W*C
            if self.sr_ratio == 0:
                self.sr_ratio = random.choice([1,2,4])
            lq_text_image, _ = bsrgan_degradation(hq_text_image, sf=self.sr_ratio, lq_patchsize=None)  #RGB 0~1 numpy h*w*c
            lq_text_image = lq_text_image.astype(np.float32)

        return lq_text_image