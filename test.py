import cv2
import sys
import os
import numpy as np
import argparse

import mser
import detect_char.model

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
for index, value in enumerate(probs):
    if value[0] - value[1] > 0.1:
        positive_boxes.append(boxes[index])

# for box in positive_boxes:
#     (x, y, w, h) = box
#
#     for box2 in positive_boxes:
#         (x2, y2, w2, h2) = box2
#
#         if x2 <= x and y2 <= y and w2 < w and h2 < h:
#             positive_boxes.remove(box2)

for box in positive_boxes:
    (x, y, w, h) = box
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

_, buf = cv2.imencode('.png', img)
print(buf.tobytes())
