import cv2
import numpy as np
import argparse

import mser
import recognize.data
from keras.models import model_from_json


def to_keras_input(images):
    result = []
    for image in images:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_image = np.expand_dims(gray_image, axis=2)
        result.append(gray_image.transpose((2, 1, 0)))

    return np.array(result)

parser = argparse.ArgumentParser()
parser.add_argument('image')
args = parser.parse_args()

img = cv2.imread(args.image)
regions = mser.detect_regions(img)
boxes = [mser.bounding_box_of_region(img, region) for region in regions]

rois = mser.mser_rois(img)
rois = np.array([cv2.resize(roi, (32, 32)) for roi in rois])

detect_model = model_from_json(open('saved_model/detect_char/20160721.json').read())
detect_model.load_weights('saved_model/detect_char/20160721.h5')
detect_probs = detect_model.predict(to_keras_input(rois))

positive_boxes = []
positive_rois = []

threshold = 0.6
for index, value in enumerate(detect_probs):
    if value[0] > threshold:
        positive_boxes.append(boxes[index])
        positive_rois.append(rois[index])

recognize_model = model_from_json(open('saved_model/recognize/20160719.json').read())
recognize_model.load_weights('saved_model/recognize/20160719-200.h5')

recognize_input = to_keras_input(positive_rois)
probs = recognize_model.predict(recognize_input)

args = np.argsort(-probs)

candidates = []
for i, box in enumerate(positive_boxes):
    candidates.append({
        'box': box,
        'prob': probs[i],
    })

candidates = sorted(candidates, key=lambda c: -1 * c['box'][2] * c['box'][3])
for c in candidates:
    (x, y, w, h) = c['box']

    for c2 in candidates:
        (x2, y2, w2, h2) = c2['box']
        contains = x < x2 and y < y2 and x2 + w2 <= x + w and y2 + h2 <= y + h
        if contains:
            c_top_prob = np.argmax(c['prob'])
            c2_top_prob = np.argmax(c2['prob'])

            if c_top_prob == c2_top_prob:
                candidates.remove(c2)

for c in candidates:
    top_prob_index = np.argmax(c['prob'])
    top_char = recognize.data.label_chars[top_prob_index]

    (x, y, w, h) = c['box']
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
