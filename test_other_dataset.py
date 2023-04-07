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


label = ['dog','horse','cat','cattle','pig',
         'orange','durian','apple','grape','banana',
         'bus','plane','train','ship','car']

# label = ['dog','horse','bus','plane','train',
#          'ship','car','cat','cattle','pig',
#          'orange','durian','apple','grape','banana']

# label = ['cat','dog']


def count_grayscale(img):
    h, w = img.shape[:2]  # template_gray 为灰度图
    gray_all = 0
    for i in range(h):
        for j in range(w):
            gray_all = gray_all + img[i, j]
    gray_average = gray_all * 1.0 / (h * w)
    return gray_average



def detect():

#     '''
#     方法3：加载图片检验准确率
#     '''

    # 加载训练好的模型。根据自己情况进行修改

    m = cc_net()
    m.load_weights("models/pdnet_1.h5") #1627



    # 设置图片路径
    # sort_number = 0
    # path = "D:/znc/ai/picture/large_data_test/" + str(sort_number)
    # filelist = os.listdir(path)
    # 设置图片路径

    # for i in range(2):
    #     os.makedirs(f"D:/znc/ai/picture/wrong_pic1/{i}", exist_ok=True)

# 批量测试图片
    all_count = 0
    for sort_number in range(2):
        path = "E:/research/parkinson_detect/pd_net/basic_data/begin/" + str(sort_number)
        # path = "E:/research/parkinson_detect/pd_net/otherdataset/data1/drawings/spiral/training/" + str(sort_number)
        # path = "E:/research/parkinson_detect/pd_net/data/test/" + str(sort_number)
        filelist = os.listdir(path)
        count = 0
        for item in filelist:
            if item.endswith(('.jpeg', 'png', 'jpg')):
                # img = cv2.imread("D:/znc/ai/picture/animal1/a0.jpg")
                # 将640*256分辨率resize成160*64。并进行BGR to RGB以及归一化
                src = os.path.join(os.path.abspath(path), item)
                # img = cv2.imread(src)
                img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)

                # ##******************************根据情况二值化***********************************##
                resize = cv2.resize(img, (227, 227))

                gray_average = count_grayscale(resize)
                # print(gray_average)

                if gray_average <= 127:
                    ret, thresh = cv2.threshold(resize, 127, 255, cv2.THRESH_BINARY)
                else:
                    ret, thresh = cv2.threshold(resize, 190, 255, cv2.THRESH_BINARY_INV)

                x = cv2.resize(thresh, (227, 227))[np.newaxis, ..., np.newaxis] / 255.0
                # ##******************************根据情况二值化***********************************##


                # ##******************************应对新的数据集***********************************##
                # thresh, new_img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY_INV)
                # img = new_img
                # ##******************************应对新的数据集***********************************##



                # x = cv2.resize(img, (227, 227))[np.newaxis, ... ,np.newaxis] / 255.0
                # print(x)

                y = np.argmax(m.predict(x)[0])
                # cv2.putText(img, str(y), (20, 20), 1, 1, (0, 255, 0), 2)
                # cv2.imshow("result", img)
                # cv2.waitKey(500)

                if y != sort_number:
                    # cv2.imwrite("D:/znc/ai/picture/wrong_pic1/" + str(sort_number) + "/" + item, img)
                    # shutil.move(path + '/' + item, "E:/znc/data_in4/wrong0/" + item)
                    count += 1
                    all_count += 1

                # cv2.putText(img, label[y], (50, 50), 3, 2, (0, 255, 0), 2)
                cv2.putText(thresh, str(y), (20, 20), 1, 1, (255, 0, 0), 1)
                cv2.imshow("result", thresh)
                cv2.waitKey(50)
        print(count)
    print("Wrong Picture:" + str(all_count))

# 批量测试图片


        # # 挑选识别错的图片
        # if y != sort_number:
        #     # 写入文件夹
        #     cv2.imwrite("E:/znc/data_in4/wrong" + str(sort_number) + "/" + item, dst)
        #     # shutil.move(path + '/' + item, "E:/znc/data_in4/wrong0/" + item)

        # cv2.waitKey(800)
        # k = cv2.waitKey(10)



if __name__ == "__main__":
    detect()



