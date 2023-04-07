from tensorflow.keras.layers import *
from tensorflow.keras.models import *


def cc_net():
    # 创建模型序列
    model = Sequential()
    # 第一层卷积网络，使用96个卷积核，大小为11x11步长为4， 要求输入的图片为227x227， 3个通道，不加边，激活函数使用relu
    model.add(Conv2D(48, (11, 11), strides=(4, 4), input_shape=(227, 227, 1), padding='same', activation='relu',
                     kernel_initializer='uniform'))
    # 池化层
    # model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # 第二层加边使用256个5x5的卷积核，加边，激活函数为relu
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same', activation='relu', kernel_initializer='uniform'))

    # 使用池化层，步长为2
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
    model.add(Dropout(0.5))
    # 第三层卷积，大小为3x3的卷积核使用384个
    model.add(Conv2D(192, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # 第四层卷积,同第三层
    model.add(Conv2D(192, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # 第五层卷积使用的卷积核为256个，其他同上
    # model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    return model

# 将padding改为具体的数值(bug)
def cc_net_new1():
    # 创建模型序列
    model = Sequential()
    # 第一层卷积网络，使用96个卷积核，大小为11x11步长为4， 要求输入的图片为227x227， 3个通道，不加边，激活函数使用relu
    model.add(Conv2D(48, (11, 11), strides=(4, 4), input_shape=(227, 227, 1), padding=2, activation='relu',
                     kernel_initializer='uniform'))
    # 池化层
    # model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # 第二层加边使用256个5x5的卷积核，加边，激活函数为relu
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding=2, activation='relu', kernel_initializer='uniform'))

    # 使用池化层，步长为2
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
    model.add(Dropout(0.5))
    # 第三层卷积，大小为3x3的卷积核使用384个
    model.add(Conv2D(192, (3, 3), strides=(1, 1), padding=1, activation='relu', kernel_initializer='uniform'))
    # 第四层卷积,同第三层
    model.add(Conv2D(192, (3, 3), strides=(1, 1), padding=1, activation='relu', kernel_initializer='uniform'))
    # 第五层卷积使用的卷积核为256个，其他同上
    # model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    return model