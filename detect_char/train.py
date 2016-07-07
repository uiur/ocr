import tensorflow as tf
import data
import time
from model import *

x = tf.placeholder(tf.float32, shape=[None, data.SIZE, data.SIZE, data.CHANNEL])
y_ = tf.placeholder(tf.float32, shape=[None, 2])
keep_prob = tf.placeholder(tf.float32)

logits = inference(x, keep_prob=keep_prob)
total_loss = loss(logits, y_)
predictions = tf.nn.softmax(logits)

train_step = train(total_loss)

batch_op = data.batch(50)
load_test_op = data.load_test()

saver = tf.train.Saver()
sess = tf.Session()

merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("./tmp_tensorflow/logs", sess.graph_def)
accuracy = evaluate(predictions, y_)

sess.run(tf.initialize_all_variables())
tf.train.start_queue_runners(sess=sess)

test_images, test_labels = sess.run(load_test_op)

start_time = time.time()

for i in range(10000):
  batch = sess.run(batch_op)

  if i % 100 == 0:
      train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0 })
      (m, test_accuracy) = sess.run([merged, accuracy], feed_dict={x: test_images, y_: test_labels, keep_prob: 1.0 })

      writer.add_summary(m, i)

      print('epoch: %d  train: %0.04f   test: %0.04f    time: %0.03f' % (i, train_accuracy, test_accuracy, time.time() - start_time))
      start_time = time.time()

  sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  if (i+1) % 1000 == 0:
     saver.save(sess, './tmp_tensorflow/train/20160707', global_step=(i + 1))

print(sess.run(accuracy, feed_dict={x: test_images, y_: test_labels, keep_prob: 1.0}))
