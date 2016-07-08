import sys
import numpy as np
import cv2
import random


def corners_of_region(region):
    top_left = np.min(region, axis=0)
    bottom_right = np.max(region, axis=0)

    return top_left, bottom_right


def bounding_boxes(img):
    mser = cv2.MSER(_min_area=10)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    regions = mser.detect(gray)


def bounding_box_of_region(img, region):
    top_left, bottom_right = corners_of_region(region)

    height, width, _ = img.shape

    margin = 6
    y_min = max(0, top_left[1] - margin)
    y_max = min(height-1, bottom_right[1]+margin)

    x_min = max(0, top_left[0]-margin)
    x_max = min(width-1, bottom_right[0]+margin)

    return (x_min, y_min, x_max - x_min, y_max - y_min)


def extract_region(img, region):
    (x, y, w, h) = bounding_box_of_region(img, region)
    region_of_image = img.copy()[y:y+h+1, x:x+w+1]

    return region_of_image


def region_gray(img, region):
    roi = extract_region(img, region)

    mask = np.zeros(roi.shape, dtype=roi.dtype)
    top_left, bottom_right = corners_of_region(region)

    relative_region = region - top_left

    cv2.drawContours(mask, [relative_region], -1, (255, 255, 255), 3)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return mask

def detect_regions(img):
    mser = cv2.MSER(_min_area=10)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return mser.detect(gray)

def mser_rois(img):
    regions = detect_regions(img)

    return [extract_region(img, region) for region in regions]

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    rois = mser_rois(img)

    i = 0
    for roi in rois:
        cv2.imwrite('tmp/%d.png' % (i), roi)
        i += 1
