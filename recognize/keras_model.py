from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils

import os
import string
import glob
import random
import numpy as np
from scipy.misc import imread, imresize

random.seed(42)
np.random.seed(42)

label_chars = string.digits + string.ascii_letters

batch_size = 32
nb_classes = len(label_chars)
nb_epoch = 200

nb_test = 1000
data_augmentation = True

img_rows, img_cols = 32, 32
img_channels = 3


def char_to_label(char):
    index = label_chars.find(char)

    label_array = np.zeros(len(label_chars), dtype=np.float32)
    label_array[index] = 1.0

    return label_array


def path_to_class_char(path):
    class_char = os.path.basename(os.path.dirname(path))

    if len(class_char) == 2 and class_char[1] == '_':
        class_char = class_char[0].upper()

    return class_char

def load_test():
    test_dirname = './data/icdar2003_test/test'
    paths = glob.glob(test_dirname + '/*/*.png')

    images = []
    labels = []
    for path in random.sample(paths, nb_test):
        images.append(read_image(path))
        labels.append(char_to_label(path_to_class_char(path)))

    return np.array(images), np.array(labels)


def read_image(path):
    image = imresize(imread(path, mode='RGB'), (img_rows, img_cols))
    image = image.transpose((2, 1, 0)).astype('float32')
    return image / 255.0


def load_data(dirname, sample_size=1024):
    paths = glob.glob(dirname + '/*/*.png')

    while True:
        sample_paths = random.sample(paths, sample_size)
        chars = [path_to_class_char(path) for path in sample_paths]

        images = None
        for sample_path in sample_paths:
            image = read_image(sample_path)
            if images is not None:
                images = np.append(images, [image], axis=0)
            else:
                images = np.array([image])

        labels = np.array([char_to_label(char) for char in chars])

        yield images, labels

def deprocess_image(image):
    image = image.transpose((1, 2, 0))
    image *= 255.0
    image = np.clip(image, 0, 255).astype('uint8')
    return image

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

x_test, y_test = load_test()

datagen = ImageDataGenerator(
    rotation_range=30,
)

epoch = 0
for x_train, y_train in load_data('./data/char74k/train'):
    if epoch > nb_epoch:
        break

    flow = datagen.flow(x_train, y_train, batch_size=batch_size)

    epoch_per_chunk = x_train.shape[0] / batch_size
    hist = model.fit_generator(
        flow,
        verbose=0,
        samples_per_epoch=batch_size,
        validation_data=(x_test, y_test),
        nb_epoch=epoch_per_chunk,
    )

    h = hist.history

    print("epoch:%d\tloss:%.4f\tacc:%.4f\tval_loss:%.4f\tval_acc:%.4f" % (epoch, h['loss'][-1], h['acc'][-1], h['val_loss'][-1], h['val_acc'][-1]))
    epoch += 1
