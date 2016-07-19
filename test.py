import cv2
import sys
import os
import numpy as np
import argparse
import string
import tensorflow as tf

import mser
import detect_char.model
import recognize.data
from keras.models import model_from_json

parser = argparse.ArgumentParser()
parser.add_argument('image')
parser.add_argument('--model', default='tmp_tensorflow/train/train/20160707-6000')
args = parser.parse_args()

path = os.path.join(os.path.dirname(__file__), args.model)

img = cv2.imread(args.image)
regions = mser.detect_regions(img)
boxes = [mser.bounding_box_of_region(img, region) for region in regions]

rois = mser.mser_rois(img)
rois = np.array([cv2.resize(roi, (32, 32)) for roi in rois])

probs = detect_char.model.predict(path, rois)

positive_boxes = []
positive_rois = []

threshold = 0.6
for index, value in enumerate(probs):
    if value[0] > threshold:
        positive_boxes.append(boxes[index])
        positive_rois.append(rois[index])

recognize_model = model_from_json(open('saved_model/recognize/20160719.json').read())
recognize_model.load_weights('saved_model/recognize/20160719-200.h5')

recognize_input = []
for roi in positive_rois:
    image = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    image = np.expand_dims(image, axis=2)
    recognize_input.append(image.transpose((2, 1, 0)))

recognize_input = np.array(recognize_input)
probs = recognize_model.predict(recognize_input)

args = np.argsort(-probs)

for i, box in enumerate(positive_boxes):
    top_char = recognize.data.label_chars[args[i][0]]

    (x, y, w, h) = box
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = .6
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(top_char, font_face, font_scale, thickness)

    contours = np.array([[x, y], [x, y - text_height], [x + text_width, y - text_height], [x + text_width, y]])
    cv2.fillPoly(img, [contours], (0, 0, 255))
    cv2.putText(img, top_char, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)


_, buf = cv2.imencode('.png', img)
print(buf.tobytes())
