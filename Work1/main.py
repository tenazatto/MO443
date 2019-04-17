import argparse
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from frequency.frequency_domain import FrequencyDomainFilter
from spatial.spatial_domain import SpatialDomainFilter

parser = argparse.ArgumentParser(description='MO443A Work 1')
parser.add_argument('-images', nargs='+', dest='images', required=True)
parser.add_argument('-spat-filter', type=int, dest='spat_filter')
parser.add_argument('-show-histogram', dest='show_histogram', action='store_true')
parser.add_argument('-freq-type', type=int, dest='freq_type')
parser.add_argument('-freq-radius', type=int, dest='freq_radius')
parser.add_argument('-freq-band', type=int, dest='freq_bandwidth')
parser.add_argument('-freq-gen-aux-images', dest='freq_gen_aux_images', action='store_true')

def apply_spatial_domain_filter(images, spat_filter, spat_histogram):
    print("Applying spatial domain filter " + str(spat_filter))
    for image in images:
        start_time = time.process_time()

        imagePaths = image.split('/')
        imageName = imagePaths[len(imagePaths) - 1]
        filterName = 'filter' + str(spat_filter)

        cvImage = cv2.imread(image, 0)

        filteredImage = SpatialDomainFilter.applyFilter(cvImage, imageName, spat_filter, spat_histogram)

        generateImage('./result_images/spatial/' + filterName + '/', imageName, filteredImage)

        print("Execution time for image %s: %s seconds" % (imageName, str(time.process_time() - start_time)))


def apply_frequency_domain_filter(images, freq_type, freq_radius, freq_bandwidth, freq_gen_aux_images, freq_histogram):
    print("Applying frequency domain filter " + str(freq_type))
    for image in images:
        start_time = time.process_time()

        imagePaths = image.split('/')
        imageProps = imagePaths[len(imagePaths) - 1].split('.')
        filterName = 'filter' + str(freq_type)

        if (freq_type > 2):
            maskName = 'mask_r' + str(freq_radius) + 'b' + str(freq_bandwidth)
            filteredImageName = imageProps[0] + '_r' + str(freq_radius) + 'b' + str(freq_bandwidth)
            magnitudeSpectrumName = imageProps[0] + 'Spectrum_r' + str(freq_radius) + 'b' + str(freq_bandwidth)
        else:
            maskName = 'mask_r' + str(freq_radius)
            filteredImageName = imageProps[0] + '_r' + str(freq_radius)
            magnitudeSpectrumName = imageProps[0] + 'Spectrum_r' + str(freq_radius)

        cvImage = cv2.imread(image, 0)
        imageMask, maskedImage, filteredImage, magnitudeSpectrum = \
            FrequencyDomainFilter.applyFilter(cvImage, filteredImageName, freq_type, freq_radius, freq_bandwidth,
                                              freq_histogram)

        if (freq_gen_aux_images):
            generateImage('./result_images/frequency/' + filterName + '/', maskName + '.' + imageProps[1], imageMask)
            generateImage('./result_images/frequency/' + filterName + '/', magnitudeSpectrumName + '.' + imageProps[1],
                          magnitudeSpectrum)

        generateImage('./result_images/frequency/' + filterName + '/', filteredImageName + '.' + imageProps[1],
                      filteredImage)

        print("Execution time for image %s: %s seconds" % (imagePaths[len(imagePaths) - 1], str(time.process_time() - start_time)))

def generateImage(folderPath, fileName, image):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    cv2.imwrite(folderPath + fileName, image)

def showHistogram(image, imageName):
    n, bins, patches = plt.hist(image.ravel(),256,[0,256])
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of ' + imageName)
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    if not os.path.exists('./result_images/histogram'):
        os.makedirs('./result_images/histogram')
    plt.savefig('./result_images/histogram/histogram_' + imageName)

def main():
    args = parser.parse_args()

    if args.spat_filter != None:
        apply_spatial_domain_filter(args.images, args.spat_filter, args.show_histogram)
    if args.freq_type != None:
        apply_frequency_domain_filter(args.images, args.freq_type, args.freq_radius, args.freq_bandwidth,
                                      args.freq_gen_aux_images, args.show_histogram)

    if (args.spat_filter is None) & (args.freq_type is None) & args.show_histogram:
        for image in args.images:
            cvImage = cv2.imread(image, 0)

            imagePaths = image.split('/')
            imageName = imagePaths[len(imagePaths) - 1]
            showHistogram(cvImage, imageName)



if __name__ == '__main__':
    main()

