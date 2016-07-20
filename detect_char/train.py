import glob
import random
import datetime
import numpy as np
import os
from scipy.misc import imread, imresize

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

random.seed(42)
np.random.seed(42)

img_rows = 32
img_cols = 32
img_channels = 1


def load_from_dir(dirname):
    positive_paths = glob.glob(dirname + '/1/*.png')
    positive_images = np.array([read_image(path) for path in positive_paths])
    positive_labels = np.array([[0., 1.] for _ in range(len(positive_paths))])

    negative_paths = glob.glob(dirname + '/0/*.png')

    negative_images = np.array([read_image(path) for path in negative_paths])
    negative_labels = np.array([[1., 0.] for _ in range(len(negative_paths))])

    return np.concatenate((positive_images, negative_images), axis=0), np.concatenate((positive_labels, negative_labels), axis=0)


def read_image(path):
    mode = 'RGB' if img_channels == 3 else 'L'
    image = imresize(imread(path, mode=mode), (img_rows, img_cols))
    if img_channels == 1:
        image = np.expand_dims(image, axis=2)

    image = image.transpose((2, 1, 0)).astype('float32')
    return image / 255.0

x_train, y_train = load_from_dir('data/char_or_not/train')
x_test, y_test = load_from_dir('data/char_or_not/test')

model = Sequential()

model.add(Convolution2D(32, 3, 3,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy'],
)

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x_train, y_train, nb_epoch=100, validation_data=(x_test, y_test), callbacks=[early_stopping])


date_string = datetime.datetime.now().strftime("%Y%m%d")
dirname = 'saved_model/detect_char'
os.makedirs(dirname, exist_ok=True)
open('%s/%s.json' % (dirname, date_string), 'w').write(model.to_json())
model.save_weights('%s/%s.h5' % (dirname, date_string))
