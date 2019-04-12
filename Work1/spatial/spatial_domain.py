import cv2

import numpy as np
from skimage.exposure import rescale_intensity


class SpatialDomainFilter:
    @staticmethod
    def applyFilter(image, spat):
        if (spat == 1):
            return SpatialDomainFilter.applyConvolution(image, SpatialDomainFilter.filter1())
        elif (spat == 2):
            return SpatialDomainFilter.applyConvolution(image, SpatialDomainFilter.filter2())
        elif (spat == 3):
            return SpatialDomainFilter.applyConvolution(image, SpatialDomainFilter.filter3())
        elif (spat == 4):
            return SpatialDomainFilter.applyConvolution(image, SpatialDomainFilter.filter4())
        elif (spat == 5):
            return SpatialDomainFilter.applyConvolution(image, SpatialDomainFilter.filter5())

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

        filteredImage = rescale_intensity(output, in_range=(0, 255))
        filteredImage = (output * 255).astype("uint8")

        return filteredImage

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

    @staticmethod
    def filter5():
        return np.sqrt(SpatialDomainFilter.filter3() * SpatialDomainFilter.filter3() +
                       SpatialDomainFilter.filter4() * SpatialDomainFilter.filter4())
