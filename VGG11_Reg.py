import numpy as np
from VGG11 import *

import keras
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import regularizers
from keras.optimizers import SGD
from keras.initializers import VarianceScaling
from keras import backend as K

def VGG_11_Reg(input_shape, num_classes):           # VGG11 modek with regularization
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.005)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=VarianceScaling(mode='fan_avg', distribution='normal'), kernel_regularizer=regularizers.l2(0.005)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=VarianceScaling(mode='fan_avg', distribution='normal'), kernel_regularizer=regularizers.l2(0.005)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=VarianceScaling(mode='fan_avg', distribution='normal'), kernel_regularizer=regularizers.l2(0.005)))
    model.add(MaxPooling2D((2,2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=VarianceScaling(mode='fan_avg', distribution='normal'), kernel_regularizer=regularizers.l2(0.005)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=VarianceScaling(mode='fan_avg', distribution='normal'), kernel_regularizer=regularizers.l2(0.005)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=VarianceScaling(mode='fan_avg', distribution='normal'), kernel_regularizer=regularizers.l2(0.005)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=VarianceScaling(mode='fan_avg', distribution='normal'), kernel_regularizer=regularizers.l2(0.005)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.005)))

    return model

if __name__ == "__main__":
    img_rows, img_cols = 32, 32
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)
    batch_size = 256
    num_classes = 10
    epochs = 5

    (X_train, y_train, X_test_original, y_test) = data_preprocess()  # get 32*32 image data
    X_train = transform_data(X_train, img_rows, img_cols)  # transform data to add 'channel' column
    X_test = transform_data(X_test_original, img_rows, img_cols)  # keep a copy of original test set
    y_train = keras.utils.to_categorical(y_train, num_classes)  # using one-hot code
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = VGG_11_Reg(input_shape, num_classes)
    sgd = SGD(lr=0.01, decay=0.1, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(X_test, y_test))

    # plot accuracy and loss
    plot_result(hist, '_Reg')           # add '_Reg' suffix

    # plot accuracy after rotation
    plot_rotate(model, X_test_original, y_test, img_rows, img_cols, '_Reg')

    # plot accuracy after blurring
    plot_blur(model, X_test_original, y_test, img_rows, img_cols, '_Reg')