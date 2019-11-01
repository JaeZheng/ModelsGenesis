#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/10/22 14:42
# @File    : Thyroid_Seg.py

from unet3d import *
import keras
# from keras import models
from utils.loss_function import dice_coef, dice_coef_loss
from utils.metrics import mean_iou
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def data_loader():
    X = []
    Y = []
    for i in range(309):
        img = cv2.imread('./datasets/thyroid/train/image/'+str(i)+'.bmp')
        mask = cv2.imread('./datasets/thyroid/train/label/'+str(i)+'.bmp')
        X.append(img)
        Y.append(mask)
    return X, Y


# prepare your own data
X, Y = data_loader()


# prepare the 3D model
input_channels, input_rows, input_cols, input_deps = 1, 64, 64, 32
num_class, activate = 2, 'sigmoid'
weight_dir = './pretrained_weights/Genesis_Chest_CT.h5'
models_genesis = unet_model_3d((input_channels, input_rows, input_cols, input_deps), batch_normalization=True)
print("Load pre-trained Models Genesis weights from {}".format(weight_dir))
models_genesis.load_weights(weight_dir)
x = models_genesis.get_layer('depth_13_relu').output
final_convolution = Conv3D(num_class, (1, 1, 1))(x)
output = Activation(activate)(final_convolution)
model = keras.models.Model(inputs=models_genesis.input, outputs=output)
model.compile(optimizer="adam", loss=dice_coef_loss, metrics=[mean_iou,dice_coef])

# train the model
model.fit(X, Y)

# data_loader()