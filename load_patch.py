#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/11/2 17:51
# @File    : load_patch.py

import numpy as np
import cv2

npy = np.load('datasets/thyroid/generated_patch/bat_32_64x64_2.npy')
print(npy.shape)

count = 0
for patch in npy:
    cv2.imwrite('datasets/tmp/'+str(count)+'.jpg', patch)
    count += 1
