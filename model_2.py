from tensorflow.keras.layers import *
from tensorflow.keras.models import *

def pd_net():
    # 创建模型序列
    model = Sequential()
    # 第一层卷积网络，使用96个卷积核，大小为11x11步长为4， 要求输入的图片为227x227， 3个通道，不加边，激活函数使用relu
    model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
    # 池化层
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 第二层加边使用256个5x5的卷积核，加边，激活函数为relu
    model.add(Conv2D(32, (3, 3), activation='relu'))
    # 使用池化层，步长为2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    return model