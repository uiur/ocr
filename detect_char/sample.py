import tensorflow as tf
import cv2
import data
import numpy as np

image_op = data.load_train_positive()

sess = tf.Session()
sess.run(tf.initialize_all_variables())
tf.train.start_queue_runners(sess=sess)

image = None
for y in range(10):
    images = []
    for x in range(10):
        sample_image, label = sess.run(image_op)
        images.append(sample_image)

    line_image = np.concatenate(images, axis=1)

    if image is not None :
        image = np.concatenate((image, line_image), axis=0)
    else:
        image = line_image

flag, buf = cv2.imencode('.png', image)
print(buf.tobytes())
