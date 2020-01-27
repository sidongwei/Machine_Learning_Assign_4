import numpy as np
from VGG11 import *

import keras
from keras.optimizers import SGD
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator


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

    model = VGG_11(input_shape, num_classes)
    sgd = SGD(lr=0.01, decay=0.1, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    datagen = ImageDataGenerator(           # data augmentation
        featurewise_center=True,            # average 0 featurewise
        rotation_range=10,                  # rotation within 10 degree
        width_shift_range=3,                # shifting within 3 pixel
        height_shift_range=3)
    datagen.fit(X_train)
    hist = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=len(X_train)/batch_size, epochs=epochs, verbose=2, validation_data=(X_test, y_test))

    # plot accuracy and loss
    plot_result(hist, '_Aug')           # add '_Aug' suffix

    # plot accuracy after rotation
    plot_rotate(model, X_test_original, y_test, img_rows, img_cols, '_Aug')

    # plot accuracy after blurring
    plot_blur(model, X_test_original, y_test, img_rows, img_cols, '_Aug')