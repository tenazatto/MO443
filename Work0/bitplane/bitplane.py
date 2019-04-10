import numpy as np


class BitPlaneSlicing:
    @staticmethod
    def bitPlaneSlicing(image):
        lstImages = []
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                lstImages.append(np.binary_repr(image[i][j], width=8))

        bitPlaneImages = []

        for i in range(0, 8):
            bitPlaneImages.append(
                (np.array([int(img[(7-i)]) for img in lstImages], dtype=np.uint8) * (2 ** i)).reshape(image.shape[0], image.shape[1])
            )

        return bitPlaneImages