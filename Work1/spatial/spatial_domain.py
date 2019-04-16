import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from skimage.exposure import rescale_intensity


class SpatialDomainFilter:
    @staticmethod
    def applyFilter(image, imageName, spat, histogram):
        if (spat == 1):
            filteredImage = SpatialDomainFilter.applyConvolution(image, SpatialDomainFilter.filter1())
        elif (spat == 2):
            filteredImage = SpatialDomainFilter.applyConvolution(image, SpatialDomainFilter.filter2())
        elif (spat == 3):
            filteredImage = SpatialDomainFilter.applyConvolution(image, SpatialDomainFilter.filter3())
        elif (spat == 4):
            filteredImage = SpatialDomainFilter.applyConvolution(image, SpatialDomainFilter.filter4())
        elif (spat == 5):
            filteredImage = SpatialDomainFilter.applyCombination(image, SpatialDomainFilter.filter3(),
                                                        SpatialDomainFilter.filter4())

        if (histogram):
            SpatialDomainFilter.showHistogram(filteredImage, spat, imageName)

        return filteredImage

    @staticmethod
    def applyConvolution(image, filter):
        (iH, iW) = image.shape[:2]
        (kH, kW) = filter.shape[:2]

        pad = (kW - 1) // 2
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                                   cv2.BORDER_REPLICATE)
        filteredImage = np.zeros((iH, iW), dtype="float32")

        for y in np.arange(pad, iH + pad):
            for x in np.arange(pad, iW + pad):
                roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

                k = (roi * filter).sum()

                filteredImage[y - pad, x - pad] = k

        filteredImage = rescale_intensity(filteredImage, in_range=(0, 255))

        filteredImage = (filteredImage * 255).astype("uint8")

        return filteredImage

    @staticmethod
    def applyCombination(image, filter1, filter2):
        output1 = SpatialDomainFilter.applyConvolution(image, filter1)
        output2 = SpatialDomainFilter.applyConvolution(image, filter2)

        filteredImage = np.hypot(output1, output2)

        filteredImage = rescale_intensity(filteredImage, in_range=(0, 255))

        filteredImage = (filteredImage * 255).astype("uint8")

        return filteredImage

    @staticmethod
    def showHistogram(image, spat, imageName):
        n, bins, patches = plt.hist(image.ravel(),256,[0,256])
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of ' + imageName + ' in filter ' + str(spat))
        plt.text(23, 45, r'$\mu=15, b=3$')
        maxfreq = n.max()
        # Set a clean upper y-axis limit.
        plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        if not os.path.exists('./result_images/spatial/histogram'):
            os.makedirs('./result_images/spatial/histogram')
        plt.savefig('./result_images/spatial/histogram/histogram_filter'+ str(spat) + '_' + imageName)

    @staticmethod
    def filter1():
        return np.array(([0, 0, -1, 0, 0],
                        [0, -1, -2, -1, 0],
                        [-1, -2, 16, -2, -1],
                        [0, -1, -2, -1, 0],
                        [0, 0, -1, 0, 0]),
                        dtype="int")

    @staticmethod
    def filter2():
        return (1 / 256) * np.array(([1, 4, 6, 4, 1],
                                    [4, 16, 24, 16, 4],
                                    [6, 24, 36, 24, 6],
                                    [4, 16, 24, 16, 4],
                                    [1, 4, 6, 4, 1]),
                                    dtype="int")

    @staticmethod
    def filter3():
        return np.array(([-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]),
                        dtype="int")

    @staticmethod
    def filter4():
        return np.array(([-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]),
                        dtype="int")
