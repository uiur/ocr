import cv2
import sys
import os
import numpy as np
import argparse
import string
import tensorflow as tf

import mser
import detect_char.model
import recognize.model
import recognize.data

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

# for box in positive_boxes:
#     (x, y, w, h) = box
#
#     for box2 in positive_boxes:
#         (x2, y2, w2, h2) = box2
#
#         if x2 <= x and y2 <= y and w2 < w and h2 < h:
#             positive_boxes.remove(box2)

tf.reset_default_graph()

probs = recognize.model.predict('tmp_tensorflow/recognize/20160708-3002', np.array(positive_rois))

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
