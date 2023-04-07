from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten,Dropout,MaxPooling2D,BatchNormalization


def Alexnet_model():
    model = Sequential()

    # 第一次卷积
    model.add(Conv2D(filters=96, input_shape=(227, 227, 3), kernel_size=(11, 11), strides=(4, 4), activation='relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 第二次卷积
    model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='same', activation='relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    # 第三次卷积
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

    # 第四次卷积
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

    # 第五次卷积
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

    # Max Pooling
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    # 扁平化，多维变一维
    model.add(Flatten())

    # 第一次全连接
    model.add(Dense(4096, activation='relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.5))

    # 第二次全连接
    model.add(Dense(4096, activation='relu'))
    # Add Dropout
    model.add(Dropout(0.5))

    # 输出层（全连接 softmax激活）
    model.add(Dense(2, activation='softmax'))
    return model


model=Alexnet_model()
# 查看模型结构
model.summary()
