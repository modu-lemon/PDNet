'''
使用电脑的USB摄像头测试模型识别效果
'''
import numpy as np
import time
from tensorflow.keras.models import load_model
import cv2
import os
import shutil
import random

from model import cc_net
from tensorflow.keras import models
from tensorflow.keras import layers

def count_grayscale(img):
    h, w = img.shape[:2]  # template_gray 为灰度图
    gray_all = 0
    for i in range(h):
        for j in range(w):
            gray_all = gray_all + img[i, j]
    gray_average = gray_all * 1.0 / (h * w)
    return gray_average

def detect():

    m = cc_net()
    m.load_weights("models/ccnet_3.h5") #1627

    # src = 'E:/research/parkinson_detect/pd_net/test_data/10.JPG'
    # src = 'E:/research/parkinson_detect/pd_net/basic_data/begin/0/1.png'
    src = 'E:/research/parkinson_detect/pd_net/otherdataset/data1/drawings/spiral/training/0/V01HE03.png'
    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    resize = cv2.resize(img, (227, 227))

    gray_average = count_grayscale(resize)
    print(gray_average)

    if gray_average <= 127:
        ret, thresh = cv2.threshold(resize, 127, 255, cv2.THRESH_BINARY)
    else:
        ret, thresh = cv2.threshold(resize, 190, 255, cv2.THRESH_BINARY_INV)

    x = cv2.resize(thresh, (227, 227))[np.newaxis, ... ,np.newaxis] / 255.0

    y = np.argmax(m.predict(x)[0])
    if y == 0:
        print('no')
    else:
        print('yes')

    cv2.putText(thresh, str(y), (20, 20), 1, 1, (255, 0, 0), 1)
    cv2.imshow("result", thresh)
    cv2.waitKey(0)


if __name__ == "__main__":
    detect()



