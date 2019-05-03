import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from dither import Dither

parser = argparse.ArgumentParser(description='MO443A Work 1')
parser.add_argument('-images', nargs='+', dest='images', required=True)
parser.add_argument('-tech', type=int, dest='tech')
parser.add_argument('-change-dir', dest='change_dir', action='store_true')


def apply_halftone_dither(images, tech, change_dir):
    print("Applying spatial domain filter ")
    for image in images:

        imagePaths = image.split('/')
        imageName = imagePaths[len(imagePaths) - 1].split('.')[0] + changeDirStr(change_dir) + '.pbm'
        ditherName = 'Bayer' if tech == 2 else 'Normal'

        cvImage = cv2.imread(image, 0)

        ditheredImage = Dither.applyHalftone(cvImage, tech, change_dir)

        generateImage('./result_images/halftone/' + ditherName + '/', imageName, ditheredImage)

def apply_floyd_steinberg_dither(images, change_dir):
    print("Applying floyd steinberg dither")
    for image in images:

        imagePaths = image.split('/')
        imageName = imagePaths[len(imagePaths) - 1].split('.')[0] + changeDirStr(change_dir) + '.pbm'

        cvImage = cv2.imread(image, 0)

        ditheredImage = Dither.applyFloydSteinberg(cvImage, change_dir)

        generateImage('./result_images/floyd_steinberg/', imageName, ditheredImage)


def changeDirStr(change_dir):
    return '-change-dir' if change_dir else '-linear'

def generateImage(folderPath, fileName, image):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    cv2.imwrite(folderPath + fileName, image)


def main():
    args = parser.parse_args()

    change_dir = args.change_dir if args.change_dir != None else False

    if args.tech == 1:
        apply_halftone_dither(args.images, args.tech, change_dir)
    elif args.tech == 2:
        apply_halftone_dither(args.images, args.tech, change_dir)
    elif args.tech == 3:
        apply_floyd_steinberg_dither(args.images, change_dir)


if __name__ == '__main__':
    main()

