import glob
import tensorflow as tf
import os
import string
import numpy as np

SIZE = 32
CHANNEL = 1
IMAGE_SHAPE = [SIZE, SIZE, CHANNEL]

data_dir = os.path.abspath(os.path.dirname(__file__) + '/../data')

label_chars = string.digits + string.ascii_letters

def normalize(image):
    return image / 255.0 - 0.5


def load_image(pattern, distort=False):
    filenames = glob.glob(pattern)
    queue = tf.train.string_input_producer(filenames)

    reader = tf.WholeFileReader()
    key, value = reader.read(queue)
    image = tf.image.decode_png(value, channels=CHANNEL)
    image = tf.image.resize_images(image, SIZE, SIZE)
    image.set_shape(IMAGE_SHAPE)
    image = tf.cast(image, tf.float32)
    if distort:
        image = tf.image.random_brightness(image, max_delta=0.4)
        image = tf.image.random_contrast(image, lower=0.6, upper=1.4)

    return image


def batch(batch_size):
    return tf.train.shuffle_batch_join(
        load_train(),
        batch_size,
        100,
        10,
        shapes=[IMAGE_SHAPE, [len(label_chars)]]
    )


def char_to_label(char):
    index = label_chars.find(char)

    label_array = np.zeros(len(label_chars), dtype=np.float32)
    label_array[index] = 1.0

    return label_array


def path_to_class_char(path):
    class_char = os.path.basename(path)

    if len(class_char) == 2 and class_char[1] == '_':
        class_char = class_char[0].upper()

    return class_char


def load_from_dir(dirname, distort=False):
    image_ops = []
    for path in glob.glob(dirname + '/*'):
        class_char = path_to_class_char(path)
        image_op = load_image(path + '/*.png', distort=distort)

        t = tf.tuple([image_op, tf.constant(char_to_label(class_char))])

        image_ops.append(t)

    return image_ops


def load_train():
    return load_from_dir(data_dir + '/char74k/train', distort=True)


def load_test():
    image_ops = load_from_dir(data_dir + '/icdar2003_test/test')
    test_size = len(glob.glob(data_dir + '/icdar2003_test/test/*/*.png'))
    batch_op = tf.train.batch_join(image_ops, test_size) # read all

    return batch_op
