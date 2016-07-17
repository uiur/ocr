import tensorflow as tf
import data
import math

input_size = data.SIZE * data.SIZE
output_size = len(data.label_chars)


def init_weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=1.0 / math.sqrt(input_size)))


def init_bias(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=1.0))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def inference(images, keep_prob=tf.constant(1.0), layer_num1=32, layer_num2=64, layer_num3=1024):

    image = tf.reshape(data.normalize(images), [-1, data.SIZE, data.SIZE, data.CHANNEL])
    W_conv1 = init_weight([5, 5, data.CHANNEL, layer_num1])
    b_conv1 = init_bias([layer_num1])

    h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv2 = init_weight([5, 5, layer_num1, layer_num2])
    b_conv2 = init_bias([layer_num2])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    size = int(data.SIZE / (2 ** 2))
    W_fc1 = init_weight([size * size * layer_num2, layer_num3])
    b_fc1 = init_bias([layer_num3])

    h_pool2_flat = tf.reshape(h_pool2, [-1, size * size * layer_num2])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = init_weight([layer_num3, output_size])
    b_fc2 = init_bias([output_size])

    return tf.matmul(h_fc1_drop, W_fc2) + b_fc2


def loss(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))


def train(total_loss, learning_rate=0.001):
    ce_summ = tf.scalar_summary("loss", total_loss)

    return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)


def evaluate(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    accuracy_summary = tf.scalar_summary("accuracy", accuracy)

    return accuracy


def correct_prediction_count(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    return tf.reduce_sum(tf.cast(correct_prediction, tf.float32))


def predict(model_path, raw_images):
    gray_images = tf.image.rgb_to_grayscale(tf.cast(tf.constant(raw_images), tf.float32))
    images = tf.image.resize_images(gray_images, data.SIZE, data.SIZE)
    logits = inference(images, keep_prob=tf.constant(1.0))  # workaround
    predictions = tf.nn.softmax(logits)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        return sess.run(predictions)
