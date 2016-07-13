import tensorflow as tf
import time

import data
import model
import datetime

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--learning-rate', type=float,
    default=0.001
)

parser.add_argument(
    '--layer-num1', type=int,
    default=32
)

parser.add_argument(
    '--layer-num2', type=int,
    default=64
)

args = parser.parse_args()
print vars(args)

TEST_SIZE = data.test_size()

date_str = datetime.datetime.now().strftime("%Y%m%d")

x = tf.placeholder(tf.float32, shape=[None, data.SIZE, data.SIZE, data.CHANNEL])
y_ = tf.placeholder(tf.float32, shape=[None, len(data.label_chars)])
keep_prob = tf.placeholder(tf.float32)

logits = model.inference(x, keep_prob=keep_prob, layer_num1=args.layer_num1, layer_num2=args.layer_num2)
total_loss = model.loss(logits, y_)
predictions = tf.nn.softmax(logits)

train_step = model.train(total_loss, learning_rate=args.learning_rate)

batch_op = data.batch(50)

saver = tf.train.Saver()
sess = tf.Session()

merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("./tmp_tensorflow/recognize/logs", sess.graph_def)

accuracy = model.evaluate(predictions, y_)
correct_prediction_count = model.correct_prediction_count(predictions, y_)

test_batch_op = data.load_test_batch(64)

sess.run(tf.initialize_all_variables())
tf.train.start_queue_runners(sess=sess)

start_time = time.time()

def eval_in_batch(sess):
    correct_count = 0
    total_test_size = 0
    while True:
        if total_test_size >= TEST_SIZE:
            break

        test_images, test_labels = sess.run(test_batch_op)

        correct_count += sess.run(correct_prediction_count, feed_dict={
          x: test_images,
          y_: test_labels,
          keep_prob: 1.0
        })
        total_test_size += len(test_images)

    test_accuracy = correct_count / total_test_size
    return test_accuracy

step = 0
prev_test_accuracy = 0.0
while True:
    batch = sess.run(batch_op)

    if step % 100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})

        test_accuracy = eval_in_batch(sess)

        print('step:%d  train:%0.04f   test:%0.04f    time:%0.03f' % (step, train_accuracy, test_accuracy, time.time() - start_time))

        start_time = time.time()

    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    if step > 0 and step % 1000 == 0:
        saver.save(sess, './tmp_tensorflow/recognize/' + date_str, global_step=step)

        # early stopping
        if step >= 2000 and abs(test_accuracy - prev_test_accuracy) < prev_test_accuracy * 0.05:
            break

        if step >= 10000:
            break

        prev_test_accuracy = test_accuracy

    step += 1
