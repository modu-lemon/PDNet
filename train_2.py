'''
使用gpu训练
'''
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.layers import *
from tensorflow.keras.models import *

import numpy as np
import os
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import cc_net
from model_2 import pd_net

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = "E:/research/parkinson_detect/pd_net/data/train"
validation_dir = "E:/research/parkinson_detect/pd_net/data/test"

epochs = 48
batch_size = 32


train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='binary')


def count_picture(path):
    count=0
    for root,dirs,files in os.walk(path):
        for each in files:
            if each.endswith('png'):
                count=count+1
    return count




if __name__ == "__main__":
    if not (os.path.exists('models')):
        os.mkdir("models")

    train_num = count_picture(train_dir)
    validation_num = count_picture(validation_dir)

    # # 创建模型序列
    # model = Sequential()
    # # 第一层卷积网络，使用96个卷积核，大小为11x11步长为4， 要求输入的图片为227x227， 3个通道，不加边，激活函数使用relu
    # model.add(Conv2D(48, (11, 11), strides=(4, 4), input_shape=(227, 227, 1), padding='same', activation='relu',
    #                  kernel_initializer='uniform'))
    # # 池化层
    # # model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # # 第二层加边使用256个5x5的卷积核，加边，激活函数为relu
    # model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same', activation='relu', kernel_initializer='uniform'))
    #
    # # 使用池化层，步长为2
    # model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
    # model.add(Dropout(0.5))
    # # 第三层卷积，大小为3x3的卷积核使用384个
    # model.add(Conv2D(192, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # # 第四层卷积,同第三层
    # model.add(Conv2D(192, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # # 第五层卷积使用的卷积核为256个，其他同上
    # # model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # model.add(Dropout(0.5))
    #
    # model.add(Flatten())
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    # # model.add(Dense(1024, activation='relu'))
    # # model.add(Dropout(0.5))
    # model.add(Dense(2, activation='softmax'))
    # opt = Adam(lr=0.001)
    # model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["acc"])
    # model.summary()

    model = pd_net()
    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["acc"])
    model.summary()



    # early_stop = EarlyStopping(patience=20)
    # reduce_lr = ReduceLROnPlateau(patience=15)
    # save_weights = ModelCheckpoint("models/model_{epoch:02d}_{val_acc:.4f}.h5",
    #                                save_best_only=True, monitor='val_acc')
    # callbacks = [save_weights, reduce_lr, early_stop]
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_num // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_num // batch_size
    )
    model.save('./models/ccnet_7.h5')

    if not (os.path.exists('models_eva')):
        os.mkdir("models_eva")
    # 模型可视化
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'ro', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig("./models_eva/model_train7.jpg")
    plt.figure()

    plt.plot(epochs, loss, 'ro', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig("./models_eva/model_validation7.jpg")
    plt.show()
    # 模型可视化



