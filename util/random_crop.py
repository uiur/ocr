import numpy as np
import cv2
import random
import argparse


def random_crop(img, size):
    width = height = size
    x = random.randint(width, img.shape[1] - width)
    y = random.randint(height, img.shape[0] - height)

    return img[y:y+height, x:x+width]

if __name__ == "__main __":
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='a path to image')
    parser.add_argument(
        '--size', type=int,
        default=32, help='crop size (default: 32)'
    )
    args = parser.parse_args()

    img = cv2.imread(args.image)
    thumb = random_crop(img, args.size)

    flag, buf = cv2.imencode('.png', thumb)
    print(buf.tobytes())
