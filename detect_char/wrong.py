import tensorflow as tf
import numpy as np
import cv2
import sys

import data
import model

saved_model = sys.argv[1]

sess = tf.Session()

test_images, test_labels = data.load_test()
predictions = tf.nn.softmax(model.inference(test_images))

wrong_prediction = tf.not_equal(tf.argmax(predictions, 1), tf.argmax(test_labels, 1))
wrong_images = tf.boolean_mask(test_images, wrong_prediction)

saver = tf.train.Saver()
saver.restore(sess, saved_model)
tf.train.start_queue_runners(sess=sess)

images = sess.run(wrong_images)
image = np.concatenate(images, axis=1)

flag, buf = cv2.imencode('.png', image)
print(buf.tobytes())
