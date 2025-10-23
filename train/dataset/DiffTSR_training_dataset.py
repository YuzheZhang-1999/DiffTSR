#!/usr/bin/python
# encoding: utf-8
import cv2
import six
import sys
import lmdb
import os, random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F

sys.path.append('../')
sys.path.append('../../')

from train.dataset.utils.real_esrgan_bsrgan_degradation import real_esrgan_degradation, bsrgan_degradation
from train.dataset.utils.alphabets import alphabet


def read_txt_file_to_int_list(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            lines = [int(line.strip()) for line in lines]
        return lines
    except Exception as e:
        print(f"Error reading the file: {e}")
        return []


## Only for training dataset generation
class DiffTSR_Training_Dataset(Dataset):
    def __init__(self, FudanVI_lmdb_folder, 
                 hq_image_list_txt, 
                 max_text_length, 
                 imgW, imgH, 
                 resize_flag=True, 
                 degrade_flag=True, 
                 max_len=-1,
                 lq_image_probability=[0.5, 0.5, 0.0]):
        super().__init__()

        self.FudanVI_lmdb_folder = FudanVI_lmdb_folder
        self.hq_image_list_txt = hq_image_list_txt
        self.max_text_length = max_text_length
        self.imgW = imgW
        self.imgH = imgH
        self.resize_flag = resize_flag
        self.degrade_flag = degrade_flag
        self.max_len = max_len
        self.lq_image_probability = lq_image_probability 

        self.alphabet = alphabet

        self.hq_image_index_list = read_txt_file_to_int_list(self.hq_image_list_txt)

        item_num_total = len(self.hq_image_index_list)
        self.nSamples = item_num_total if (max_len < 0 or max_len > item_num_total) else max_len

        self.env = lmdb.open(
                    self.FudanVI_lmdb_folder,
                    max_readers=1,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False)

        if not self.env:
            print('cannot create lmdb from %s' % (self.FudanVI_lmdb_folder))
            sys.exit(0)

    def get_index_item_from_lmdb(self, lmdb_index):
        lmdb_index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % lmdb_index
            imgbuf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('RGB')
            except IOError:
                print('Corrupted image for %d' % lmdb_index)
                return self[lmdb_index + 1]

            label_key = 'label-%09d' % lmdb_index
            label = str(txn.get(label_key.encode()).decode('utf-8'))

            label_ocr = strQ2B(label)
            label_ocr += '$'
            label_ocr = label_ocr.lower()

            if len(label) <= 0:
                return self[lmdb_index + 1]

            label = label.lower()
        return (img, label, label_ocr)

    def random_blocks_masked(self, image, n_h, n_w, p):
        image = image.copy()
        h, w, c = image.shape
        block_h = h // n_h
        block_w = w // n_w

        for i in range(n_h):
            for j in range(n_w):
                if np.random.rand() < p:
                    image[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w, :] = 0
        return image

    def random_region_masked(self, image, mask_h, mask_w):
        image = image.copy()
        h, w, c = image.shape
        start_row = random.randint(0, h - mask_h)
        start_col = random.randint(0, w - mask_w)
        image[start_row:start_row+mask_h, start_col:start_col+mask_w, :] = 0
        return image

    def get_padding_label_from_text(self, text_str, max_length):
        padding_size = max(max_length - len(text_str), 0)
        check_text = ''
        label = []
        for i in text_str:
            index = self.alphabet.find(i)
            if index >= 0:
                check_text = check_text + i
                label.append(index)
            else:
                label.append(6735) 

        label = torch.tensor(label)
        if len(text_str) > max_length:
            pad_label = label[:max_length]
        else:
            pad_label = F.pad(label, (0, padding_size), value=6735)
        return check_text, label, pad_label

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        lmdb_index = self.hq_image_index_list[index]
        hq_text_image, text_str, label_ocr = self.get_index_item_from_lmdb(lmdb_index)
        
        if self.resize_flag:
            hq_text_image = hq_text_image.resize((self.imgW, self.imgH), Image.BICUBIC)

        hq_text_image = np.asarray(hq_text_image) / 255.0

        if self.degrade_flag:
            try:
                degradation_type = random.random()
                if degradation_type < self.lq_image_probability[0]: # real-esrgan
                    # input should be BGR 0~1 numpy H*W*C
                    # output is RGB 0~1 tensor
                    lq_text_image = real_esrgan_degradation(hq_text_image[:,:,::-1], insf=random.choice([1,2,4])).squeeze(0).detach().numpy() # output numpy c*h*w 0~1 RGB
                    lq_text_image = lq_text_image.transpose((1,2,0)) # transfer to h*w*c

                elif degradation_type < sum(self.lq_image_probability[:2]): # bsrgan
                    # input should be RGB 0~1 numpy H*W*C
                    # output is RGB 0~1 numpy H*W*C
                    lq_text_image, _ = bsrgan_degradation(hq_text_image, sf=random.choice([1,2,4]), lq_patchsize=None)#RGB 0~1 numpy h*w*c
                    lq_text_image = lq_text_image.astype(np.float32)
                    
                else:
                    lq_text_image = hq_text_image

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(['error degradation', hq_text_image.shape, e, exc_type, fname, exc_tb.tb_lineno])
                lq_text_image = np.ascontiguousarray(hq_text_image) # out RGB
        else:
            lq_text_image = np.ascontiguousarray(hq_text_image)

        h_lq, w_lq = lq_text_image.shape[:2]
        lq_text_image = cv2.resize(lq_text_image, (0,0), fx=self.imgW//w_lq, fy=self.imgH//h_lq,  interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]))
   
        text, label, pad_label = self.get_padding_label_from_text(text_str, self.max_text_length)

        hq_text_image = hq_text_image.copy()
        lq_text_image = lq_text_image.copy()

        hq_text_image = transforms.ToTensor()(hq_text_image)
        hq_text_image = hq_text_image.transpose(0, 1).transpose(1, 2)

        lq_text_image = transforms.ToTensor()(lq_text_image)
        lq_text_image = lq_text_image.transpose(0, 1).transpose(1, 2)

        label = torch.tensor(label, dtype=torch.long)
        pad_label = torch.tensor(pad_label, dtype=torch.long)

        example = dict()
        example["hq_image"] = hq_text_image
        example["lq_image"] = lq_text_image
        example['text_label'] = pad_label
        example['label_ocr'] = label_ocr

        return example


def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring