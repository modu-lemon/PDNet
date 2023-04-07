from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten,Dropout,MaxPool2D,BatchNormalization


# vgg16模型定义
def vgg16_original_model():
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), input_shape=(224, 224, 3), padding='same',
                     activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                     kernel_initializer='glorot_uniform'))
    model.add(MaxPool2D())  # pool_size=(2,2) strides=(2,2)

    model.add(
        Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(
        Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPool2D())

    model.add(
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(
        Conv2D(256, (1, 1), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPool2D())

    model.add(
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(
        Conv2D(512, (1, 1), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPool2D())

    model.add(
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(
        Conv2D(512, (1, 1), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPool2D())

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    return model

# vgg16原始模型定义
def vgg16_original_model():
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), input_shape=(224, 224, 3), padding='same',
                     activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                     kernel_initializer='glorot_uniform'))
    model.add(MaxPool2D())  # pool_size=(2,2) strides=(2,2)

    model.add(
        Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(
        Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPool2D())

    model.add(
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(
        Conv2D(256, (1, 1), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPool2D())

    model.add(
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(
        Conv2D(512, (1, 1), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPool2D())

    model.add(
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(
        Conv2D(512, (1, 1), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPool2D())

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    return model

# vgg16模型定义
def vgg16_model():
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), input_shape=(227, 227, 3), padding='same',
                     activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                     kernel_initializer='glorot_uniform'))
    model.add(MaxPool2D())  # pool_size=(2,2) strides=(2,2)

    model.add(
        Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(
        Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPool2D())

    model.add(
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(
        Conv2D(256, (1, 1), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPool2D())

    model.add(
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(
        Conv2D(512, (1, 1), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPool2D())

    model.add(
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(
        Conv2D(512, (1, 1), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPool2D())

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(1000, activation='relu'))
    #
    # model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    return model


model=vgg16_original_model()
# 查看模型结构
model.summary()