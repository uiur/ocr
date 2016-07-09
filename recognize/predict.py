import model
import cv2
import sys
import tensorflow as tf

image = cv2.imread(sys.argv[1])
probs = model.predict('tmp_tensorflow/recognize/20160708-3002', image)[0]

print probs
