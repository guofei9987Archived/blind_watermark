# -*- coding: utf-8 -*-
# 水印图像傅里叶变换，图像傅里叶电环相减，解码后得到水印
import cv2
import numpy as np
import random
import math



seed = 2019




ori_path = '1.jpg'
im_path = 'result.jpg' # encode 阶段生成的结果
res_path = 'mytest1.jpg'
alpha = 15 # default混杂强度



ori = cv2.imread(ori_path) / 255
im = cv2.imread(im_path) / 255
im_height, im_width, im_channel = np.shape(ori)
# # 执行一次剪切攻击（不成功）
# cut_left=int(0.8*im_width)
# im[:,:cut_left,:]=im[:,[int(0.8*i) for i in range(cut_left)],:]
# 源图像与水印图像傅里叶变换
ori_f = np.fft.fft2(ori)
ori_f = np.fft.fftshift(ori_f)
im_f = np.fft.fft2(im)
im_f = np.fft.fftshift(im_f)
mark = np.abs((im_f - ori_f) / alpha)
res = np.zeros(ori.shape)

# 获取随机种子
x, y = list(range(math.floor(im_height / 2))), list(range(im_width))
random.seed(seed)
random.shuffle(x)
random.shuffle(y)
for i in range(math.floor(im_height / 2)):
    for j in range(im_width):
        res[x[i]][y[j]] = mark[i][j] * 255
        res[im_height - i - 1][im_width - j - 1] = res[i][j]
cv2.imwrite(res_path, res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
