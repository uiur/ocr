import glob
import tensorflow as tf
import os

SIZE = 32
IMAGE_SHAPE = [SIZE, SIZE, 1]
CHANNEL = 1

data_dir = os.path.abspath(os.path.dirname(__file__) + '/../data')


def normalize(image):
    return image / 255.0 - 0.5


def load_image(pattern, distort=False):
    filenames = glob.glob(pattern)
    queue = tf.train.string_input_producer(filenames)

    reader = tf.WholeFileReader()
    key, value = reader.read(queue)
    image = tf.image.decode_png(value, channels=1)
    image = tf.image.resize_images(image, SIZE, SIZE)
    image.set_shape(IMAGE_SHAPE)
    image = tf.cast(image, tf.float32)
    if distort:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.4)
        image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
        image = tf.image.random_hue(image, max_delta=0.04)
        image = tf.image.random_saturation(image, lower=0.6, upper=1.4)

    return normalize(image)


def load_train_positive():
    image = load_image(data_dir + '/char_or_not/train/1/*.png')
    return tf.tuple([image, tf.constant([1., 0.])])


def load_train_negative():
    image = load_image(data_dir + '/char_or_not/train/0/*.png')
    return tf.tuple([image, tf.constant([0., 1.])])


def load_test():
    batch_size = len(glob.glob(data_dir + '/char_or_not/test/*/*.png'))
    positive = load_image(data_dir + '/char_or_not/test/1/*.png')
    negative = load_image(data_dir + '/char_or_not/test/0/*.png')

    return tf.train.batch_join([
        tf.tuple([positive, tf.constant([1., 0.])]),
        tf.tuple([negative, tf.constant([0., 1.])])
    ], batch_size, shapes=[IMAGE_SHAPE, [2]])


def batch(batch_size):
    return tf.train.shuffle_batch_join([load_train_positive(), load_train_negative()], batch_size, 10000, 10, enqueue_many=False, shapes=[IMAGE_SHAPE, [2]])
