import json
import cv2
import xml.etree.ElementTree as ElementTree
import os

import mser

data_dir = '/Users/zat/Downloads/svt1/'
root = ElementTree.parse(data_dir + 'train.xml').getroot()

images = [image.find('imageName').text for image in root]

images = images[0:2]

for image in images:
    img = cv2.imread(data_dir + image)
    rois = mser.mser_rois(img)

    base, ext = os.path.splitext(os.path.basename(image))

    i = 0
    for roi in rois:
        cv2.imwrite('tmp2/%s_%d.png' % (base, i), roi)
        i += 1
