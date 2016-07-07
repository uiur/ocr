import model
import cv2
import sys
import tensorflow as tf

image = cv2.imread(sys.argv[1])
prob_positive, prob_negative = model.predict(image)[0]

print "%.2f %.2f" % (prob_positive, prob_negative)
