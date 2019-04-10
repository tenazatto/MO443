import argparse
import os

import cv2
import numpy as np

from bitplane.bitplane import BitPlaneSlicing
from combination.combination import ImageCombination
from gamma.gamma import Gamma
from mosaic.mosaic import Mosaic

parser = argparse.ArgumentParser(description='MO443A Work 0')
parser.add_argument('-images', nargs='+', dest='images', required=True)
parser.add_argument('-gamma', nargs='+', type=float, dest='gamma')
parser.add_argument('-bitplane', dest='bitplane', action='store_true')
parser.add_argument('-mosaic', dest='mosaic', action='store_true')
parser.add_argument('-combine', nargs=2, type=float, dest='combine')

def apply_gamma(gammas, images):
    for gamma in gammas:
        for image in images:
            cvImage = cv2.imread(image)
            gammaImage = Gamma.adjust_gamma(cvImage, gamma)

            imagePaths = image.split('/')
            imageProps = imagePaths[len(imagePaths) - 1].split('.')
            if not os.path.exists('./result_images/gamma'):
                os.makedirs('./result_images/gamma')
            cv2.imwrite('./result_images/gamma/' + imageProps[0] + str(gamma).replace('.', 'dot') + '.'
                        + imageProps[1], np.array(gammaImage))


def apply_bitplanes(images):
    for image in images:
        cvImage = cv2.imread(image, 0)
        bitPlaneImages = BitPlaneSlicing.bitPlaneSlicing(cvImage)

        if not os.path.exists('./result_images/bitplaneslicing'):
            os.makedirs('./result_images/bitplaneslicing')
        for i in range(0, 8):
            imagePaths = image.split('/')
            imageProps = imagePaths[len(imagePaths) - 1].split('.')
            if not os.path.exists('./result_images/bitplaneslicing'):
                os.makedirs('./result_images/bitplaneslicing')
            cv2.imwrite('./result_images/bitplaneslicing/' + imageProps[0] + 'Bitplane' + str(i) + '.'
                        + imageProps[1], np.array(bitPlaneImages[i]))


def apply_mosaic(images):
    for image in images:
        cvImage = cv2.imread(image)
        mosaicImage = Mosaic.applyMosaic(cvImage)

        imagePaths = image.split('/')
        imageName = imagePaths[len(imagePaths) - 1]
        if not os.path.exists('./result_images/mosaic'):
            os.makedirs('./result_images/mosaic')
        cv2.imwrite('./result_images/mosaic/' + imageName, mosaicImage)


def apply_combination(combine, images):
    if len(images) != 2:
        print('Invalid number of images! Put only 2 of them')
    else:
        cvImage1 = cv2.imread(images[0], 0)
        cvImage2 = cv2.imread(images[1], 0)

        if cvImage1.shape[0] != cvImage2.shape[0] | cvImage1.shape[1] != cvImage2.shape[1]:
            print('Only images with same width and height are accepted')
        else:
            combineImage = ImageCombination.combineImages(combine, cvImage1, cvImage2)

            imagePaths1 = images[0].split('/')
            imageProps1 = imagePaths1[len(imagePaths1) - 1].split('.')
            imagePaths2 = images[1].split('/')
            imageProps2 = imagePaths2[len(imagePaths2) - 1].split('.')
            if not os.path.exists('./result_images/imagecombination'):
                os.makedirs('./result_images/imagecombination')
            cv2.imwrite('./result_images/imagecombination/combine' + imageProps1[0]
                        + str(combine[0]).replace('.', 'dot') + imageProps2[0] + str(combine[1]).replace('.', 'dot')
                        + '.' + imageProps1[1], np.array(combineImage))

def main():
    args = parser.parse_args()

    if args.gamma != None:
        apply_gamma(args.gamma, args.images)
    if args.bitplane:
        apply_bitplanes(args.images)
    if args.mosaic:
        apply_mosaic(args.images)
    if args.combine != None:
        apply_combination(args.combine, args.images)


if __name__ == '__main__':
    main()

