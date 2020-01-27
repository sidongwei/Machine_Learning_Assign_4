import numpy as np
from PIL import Image
from PIL.ImageFilter import GaussianBlur
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.initializers import VarianceScaling
from keras.datasets import mnist
from keras import backend as K
from keras.callbacks import LearningRateScheduler


def VGG_11(input_shape, num_classes):           # VGG11 modek, input_shape is a 4D vector, num_classes is an integer
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=VarianceScaling(mode='fan_avg', distribution='normal')))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=VarianceScaling(mode='fan_avg', distribution='normal')))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=VarianceScaling(mode='fan_avg', distribution='normal')))
    model.add(MaxPooling2D((2,2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=VarianceScaling(mode='fan_avg', distribution='normal')))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=VarianceScaling(mode='fan_avg', distribution='normal')))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=VarianceScaling(mode='fan_avg', distribution='normal')))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=VarianceScaling(mode='fan_avg', distribution='normal')))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def resize(X):          # resize a group of images into 32*32 image
    n = len(X)
    X_s = np.zeros((n,32,32))
    for i in range(n):
        im = Image.fromarray(X[i])
        X_s[i] = np.array(im.resize((32,32)))
    return X_s


def rotate(X, deg):         # rotate the image with certain degree
    n = len(X)
    X_s = np.zeros((n, 32, 32))
    for i in range(n):
        im = Image.fromarray(X[i])
        X_s[i] = np.array(im.rotate(angle=deg))

    return X_s


def blur(X, radius):        # blur the image with Gaussian filter of certain radius
    n = len(X)
    X_s = np.zeros((n, 32, 32))
    for i in range(n):
        X_s[i] = gaussian_filter(X[i], sigma=radius)
    return X_s


def transform_data(X, img_rows, img_cols):      # transform the data into 4D vector to put into the network
    if K.image_data_format() == 'channels_first':           # add the number of channels at proper position
        X_s = X.reshape(X.shape[0], 1, img_rows, img_cols)
    else:
        X_s = X.reshape(X.shape[0], img_rows, img_cols, 1)
    return X_s


def data_preprocess():          # get training and testing data
    (X_train, y_train), (X_test_original, y_test) = mnist.load_data()
    X_train = resize(X_train)
    X_test_original = resize(X_test_original)
    a = np.mean(X_train, axis=0)            # minus average as in the paper
    X_train -= a
    X_test_original -= a
    return X_train, y_train, X_test_original, y_test


def plot_fig(x, y, picname):            # plot a figure with proper name and title
    plt.title(picname)
    plt.plot(x, y)
    plt.savefig(picname+".png")
    plt.cla()


def plot_result(hist, suffix=""):            # plot the normal result w.r.t. epochs, suffix is to be added to the end
    x = [1, 2, 3, 4, 5]
    y1 = hist.history['val_acc']
    plot_fig(x, y1, "test_accuracy"+suffix)
    y2 = hist.history['acc']
    plot_fig(x, y2, "train_accuracy"+suffix)
    y3 = hist.history['val_loss']
    plot_fig(x, y3, "test_loss"+suffix)
    y4 = hist.history['loss']
    plot_fig(x, y4, "train_loss"+suffix)


def plot_rotate(model, X_test_original, y_test, img_rows, img_cols, suffix=""):            # plot result with rotation
    degree = np.arange(-45, 50, 5)
    score = np.zeros((19, 2))
    for i in range(19):
        Test = rotate(X_test_original, degree[i])
        Test = transform_data(Test, img_rows, img_cols)
        score[i] = model.evaluate(Test, y_test, verbose=0)
    plot_fig(degree, score[:, 1], "accuracy_with_rotation_degree"+suffix)


def plot_blur(model, X_test_original, y_test, img_rows, img_cols, suffix=""):         # plot result with blur
    radius = np.arange(0, 7, 1)
    score = np.zeros((7, 2))
    for i in range(7):
        Test = blur(X_test_original, radius[i])
        Test = transform_data(Test, img_rows, img_cols)
        score[i] = model.evaluate(Test, y_test, verbose=0)
    plot_fig(radius, score[:, 1], "accuracy_with_blur_radius"+suffix)


if __name__ == "__main__":
    img_rows, img_cols = 32, 32
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)
    batch_size = 256
    num_classes = 10
    epochs = 5

    (X_train, y_train, X_test_original, y_test) = data_preprocess()         # get 32*32 image data
    X_train = transform_data(X_train, img_rows, img_cols)           # transform data to add 'channel' column
    X_test = transform_data(X_test_original, img_rows, img_cols)            # keep a copy of original test set
    y_train = keras.utils.to_categorical(y_train, num_classes)          # using one-hot code
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = VGG_11(input_shape, num_classes)
    sgd = SGD(lr=0.01, decay=0.1, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(X_test, y_test))

    # plot accuracy and loss
    plot_result(hist)

    # plot accuracy after rotation
    plot_rotate(model, X_test_original, y_test, img_rows, img_cols)

    # plot accuracy after blurring
    plot_blur(model, X_test_original, y_test, img_rows, img_cols)