import argparse
import os

import cv2

from frequency.frequency_domain import FrequencyDomainFilter
from spatial.spatial_domain import SpatialDomainFilter

parser = argparse.ArgumentParser(description='MO443A Work 1')
parser.add_argument('-images', nargs='+', dest='images', required=True)
parser.add_argument('-spat', type=int, dest='spat')
parser.add_argument('-freq-type', type=int, dest='freq_type')
parser.add_argument('-freq-radius', type=int, dest='freq_radius')
parser.add_argument('-freq-band', type=int, dest='freq_bandwidth')

def apply_spatial_domain_filter(images, spat):
    for image in images:
        cvImage = cv2.imread(image)
        filteredImage = SpatialDomainFilter.applyFilter(cvImage, spat)

        imagePaths = image.split('/')
        imageName = imagePaths[len(imagePaths) - 1]
        if not os.path.exists('./result_images/spatial'):
            os.makedirs('./result_images/spatial')
        cv2.imwrite('./result_images/spatial/' + imageName, filteredImage)


def apply_frequency_domain_filter(images, freq_type, freq_radius, freq_bandwidth):
    for image in images:
        cvImage = cv2.imread(image, 0)
        imageMask, maskedImage, filteredImage, magnitudeSpectrum = \
            FrequencyDomainFilter.applyFilter(cvImage, freq_type, freq_radius, freq_bandwidth)

        imagePaths = image.split('/')
        imageName = imagePaths[len(imagePaths) - 1]
        if not os.path.exists('./result_images/frequency'):
            os.makedirs('./result_images/frequency')
        cv2.imwrite('./result_images/frequency/' + imageName, imageMask)

def main():
    args = parser.parse_args()

    if args.spat != None:
        apply_spatial_domain_filter(args.images, args.spat)
    if args.freq_type != None:
        apply_frequency_domain_filter(args.images, args.freq_type, args.freq_radius, args.freq_bandwidth)


if __name__ == '__main__':
    main()

