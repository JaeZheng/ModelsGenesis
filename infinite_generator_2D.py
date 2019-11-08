#!/usr/bin/env python
# coding: utf-8

"""
for subset in `seq 0 9`
do
python -W ignore infinite_generator_3D.py \
--fold $subset \
--scale 32 \
--data /mnt/dataset/shared/zongwei/LUNA16 \
--save generated_cubes
done
"""

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import os
import keras
print("Keras = {}".format(keras.__version__))
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from tqdm import tqdm
from sklearn import metrics
from optparse import OptionParser
from glob import glob
from skimage.transform import resize
import cv2

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("--fold", dest="fold", help="fold of subset", default=None, type="int")
parser.add_option("--input_rows", dest="input_rows", help="input rows", default=64, type="int")
parser.add_option("--input_cols", dest="input_cols", help="input cols", default=64, type="int")
parser.add_option("--input_deps", dest="input_deps", help="input deps", default=1, type="int")
parser.add_option("--crop_rows", dest="crop_rows", help="crop rows", default=64, type="int")
parser.add_option("--crop_cols", dest="crop_cols", help="crop cols", default=64, type="int")
parser.add_option("--data", dest="data", help="the directory of LUNA16 dataset", default=None, type="string")
parser.add_option("--save", dest="save", help="the directory of processed 3D cubes", default=None, type="string")
parser.add_option("--scale", dest="scale", help="scale of the generator", default=32, type="int")
(options, args) = parser.parse_args()
fold = options.fold

seed = 1
random.seed(seed)

assert options.data is not None
assert options.save is not None
assert options.fold >= 0 and options.fold <= 9

if not os.path.exists(options.save):
    os.makedirs(options.save)


class setup_config():
    hu_max = 255.0
    hu_min = 0.0
    HU_thred = (50 - hu_min) / (hu_max - hu_min)
    def __init__(self, 
                 input_rows=None, 
                 input_cols=None,
                 input_deps=None,
                 crop_rows=None, 
                 crop_cols=None,
                 len_border=None,
                 len_border_z=None,
                 scale=None,  # 单张图片生成的patch数量
                 DATA_DIR=None,
                 train_fold=[0,1,2,3,4],
                 valid_fold=[5,6],
                 test_fold=[7,8,9],
                 len_depth=None,
                 lung_min=0.7,
                 lung_max=1.0,
                ):
        self.input_rows = input_rows  # 输入图片的宽度
        self.input_cols = input_cols  # 输入图片的高度
        self.input_deps = input_deps  # 输入图片的深度，2D图像可去掉
        self.crop_rows = crop_rows
        self.crop_cols = crop_cols
        self.len_border = len_border  # 剪裁的边界，这里设为长和宽中比较小的值
        self.len_border_z = len_border_z
        self.scale = scale
        self.DATA_DIR = DATA_DIR
        self.train_fold = train_fold
        self.valid_fold = valid_fold
        self.test_fold = test_fold
        self.len_depth = len_depth
        self.lung_min = lung_min
        self.lung_max = lung_max

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


config = setup_config(input_rows=options.input_rows,
                      input_cols=options.input_cols,
                      input_deps=options.input_deps,
                      crop_rows=options.crop_rows,
                      crop_cols=options.crop_cols,
                      scale=options.scale,
                      len_border=0,
                      len_border_z=30,
                      len_depth=3,
                      lung_min=0.2,
                      lung_max=0.7,
                      DATA_DIR=options.data,
                     )
config.display()


def infinite_generator_from_one_slice(config, img_array):
    size_x, size_y = img_array.shape

    # 调整在额定的像素值范围内,并归一化到[0,1]之间
    img_array[img_array < config.hu_min] = config.hu_min
    img_array[img_array > config.hu_max] = config.hu_max
    img_array = 1.0*(img_array-config.hu_min) / (config.hu_max-config.hu_min)
    
    slice_set = np.zeros((config.scale, config.input_rows, config.input_cols), dtype=float)
    
    num_pair = 0
    cnt = 0
    while True:
        cnt += 1
        # 裁剪一定次数后，便退出
        if cnt > 50 * config.scale and num_pair == 0:
            return None
        elif cnt > 50 * config.scale and num_pair > 0:
            return np.array(slice_set[:num_pair])

        # patch的起始x坐标和y坐标
        start_x = random.randint(0+config.len_border, size_x-config.crop_rows-1-config.len_border)
        start_y = random.randint(0+config.len_border, size_y-config.crop_cols-1-config.len_border)
        # patch窗口
        crop_window = img_array[start_x : start_x+config.crop_rows,
                                start_y : start_y+config.crop_cols,
                               ]
        # if config.crop_rows != config.input_rows or config.crop_cols != config.input_cols:
        #     crop_window = resize(crop_window,
        #                          (config.input_rows, config.input_cols, config.input_deps+config.len_depth),
        #                          preserve_range=True,
        #                         )
        
        # t_img = np.zeros((config.input_rows, config.input_cols, config.input_deps), dtype=float)
        # d_img = np.zeros((config.input_rows, config.input_cols, config.input_deps), dtype=float)
        #
        # for d in range(config.input_deps):
        #     for i in range(config.input_rows):
        #         for j in range(config.input_cols):
        #             for k in range(config.len_depth):
        #                 if crop_window[i, j, d+k] >= config.HU_thred:
        #                     t_img[i, j, d] = crop_window[i, j, d+k]
        #                     d_img[i, j, d] = k
        #                     break
        #                 if k == config.len_depth-1:
        #                     d_img[i, j, d] = k
        #
        # d_img = d_img.astype('float32')
        # d_img /= (config.len_depth - 1)
        # d_img = 1.0 - d_img
        
        if np.sum(crop_window) < config.lung_min * config.input_rows * config.input_cols * config.input_deps:
            continue
        
        slice_set[num_pair] = crop_window[:,:]
        
        num_pair += 1
        if num_pair == config.scale:
            break
            
    return np.array(slice_set)


def get_self_learning_data(fold, config):
    slice_set = []
    for index_subset in fold:
        luna_subset_path = os.path.join(config.DATA_DIR, "subset/"+str(index_subset))
        file_list = glob(os.path.join(luna_subset_path, "*.bmp"))
        
        for img_file in tqdm(file_list):
            
            itk_img = sitk.ReadImage(img_file) 
            img_array = sitk.GetArrayFromImage(itk_img)
            # print("img_array shape", img_array.shape)  # (400, 500, 3)
            img_array = img_array.transpose(2, 1, 0)
            # print('max', np.max(img_array))
            # print('min', np.min(img_array))
            # print("img_array shape after transpose", img_array.shape) # (3, 500, 400)
            
            x = infinite_generator_from_one_slice(config, img_array[0])
            if x is not None:
                slice_set.extend(x)
            
    return np.array(slice_set)


print(">> Fold {}".format(fold))
cube = get_self_learning_data([fold], config)
print("cube: {} | {:.2f} ~ {:.2f}".format(cube.shape, np.min(cube), np.max(cube)))
np.save(os.path.join(options.save, 
                     "bat_"+str(config.scale)+
                     "_"+str(config.input_rows)+
                     "x"+str(config.input_cols)+
                     "_"+str(fold)+".npy"), 
        cube,
       )