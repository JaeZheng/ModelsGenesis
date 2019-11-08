#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/11/2 17:51
# @File    : load_patch.py

import numpy as np
import cv2
import copy
from scipy.misc import comb
import random
import matplotlib.pyplot as plt


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       Control points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def data_augmentation(x, y, prob=0.5):
    # augmentation by flipping
    cnt = 3
    while random.random() < prob and cnt > 0:
        degree = random.choice([0, 1, 2])
        x = np.flip(x, axis=degree)
        y = np.flip(y, axis=degree)
        cnt = cnt - 1

    return x, y


def nonlinear_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x


def local_pixel_shuffling(x, prob=0.5):
    if random.random() >= prob:
        return x
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    img_rows, img_cols = x.shape
    num_block = 500
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows//10)
        block_noise_size_y = random.randint(1, img_cols//10)
        noise_x = random.randint(0, img_rows-block_noise_size_x)
        noise_y = random.randint(0, img_cols-block_noise_size_y)
        window = orig_image[noise_x:noise_x+block_noise_size_x,
                            noise_y:noise_y+block_noise_size_y]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((block_noise_size_x,
                                 block_noise_size_y))
        image_temp[noise_x:noise_x+block_noise_size_x,
                   noise_y:noise_y+block_noise_size_y] = window
    local_shuffling_x = image_temp

    return local_shuffling_x


def image_in_painting(x):
    in_painting_x = copy.deepcopy(x)
    img_rows, img_cols = x.shape
    block_noise_size_x = random.randint(10, 20)
    block_noise_size_y = random.randint(10, 20)
    noise_x = random.randint(3, img_rows-block_noise_size_x-3)
    noise_y = random.randint(3, img_cols-block_noise_size_y-3)
    in_painting_x[noise_x:noise_x+block_noise_size_x,
      noise_y:noise_y+block_noise_size_y] = random.random()
    return in_painting_x


def image_out_painting(x):
    img_rows, img_cols = x.shape
    block_noise_size_x = img_rows - random.randint(10, 20)
    block_noise_size_y = img_cols - random.randint(10, 20)
    noise_x = random.randint(3, img_rows-block_noise_size_x-3)
    noise_y = random.randint(3, img_cols-block_noise_size_y-3)
    image_temp = copy.deepcopy(x)
    out_painting_x = copy.deepcopy(x)
    out_painting_x[:, :] = random.random()
    out_painting_x[noise_x:noise_x+block_noise_size_x,
                   noise_y:noise_y+block_noise_size_y] = image_temp[noise_x:noise_x+block_noise_size_x,
                                                                    noise_y:noise_y+block_noise_size_y]
    return out_painting_x


def load_and_write_patch():
    npy = np.load('datasets/thyroid/generated_patch/bat_32_64x64_2.npy')
    print(npy.shape)

    count = 0
    for patch in npy:
        # cv2.imwrite('datasets/tmp/'+str(count)+'.jpg', patch)
        plt.imsave('datasets/tmp/'+str(count)+'.jpg', patch, cmap='gray')
        count += 1


def transform_demo():
    npy = np.load('datasets/thyroid/generated_patch/bat_32_64x64_2.npy')
    print(npy.shape)
    # 原图
    example = npy[99]
    # 非线性变换
    non = nonlinear_transformation(example)
    # 局部像素打乱
    local_shuff = local_pixel_shuffling(example)
    # 向内补丁
    in_painting = image_in_painting(example)
    # 向外补丁
    out_painting = image_out_painting(example)
    print(np.mean(example))
    print(np.mean(non))
    print(np.mean(local_shuff))
    print(np.mean(in_painting))
    print(np.mean(out_painting))
    plt.figure(figsize=(10, 5))  # 设置窗口大小
    plt.suptitle('Multi_Image')  # 图片名称
    plt.subplot(2, 3, 1), plt.title('original')
    plt.imshow(example, cmap='gray'), plt.axis('off')
    plt.subplot(2, 3, 2), plt.title('nonlinear')
    plt.imshow(non, cmap='gray'), plt.axis('off')
    plt.subplot(2, 3, 3), plt.title('local_shuff')
    plt.imshow(local_shuff, cmap='gray'), plt.axis('off')
    plt.subplot(2, 3, 4), plt.title('in_painting')
    plt.imshow(in_painting, cmap='gray'), plt.axis('off')
    plt.subplot(2, 3, 5), plt.title('out_painting')
    plt.imshow(out_painting, cmap='gray'), plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # load_and_write_patch()
    transform_demo()