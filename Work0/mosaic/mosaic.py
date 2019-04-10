import cv2


class Mosaic:
    @staticmethod
    def applyMosaic(image):
        widthPart = int(image.shape[0] / 4)
        heightPart = int(image.shape[1] / 4)
        crops = []
        mosaic = []

        for i in range(0, 4):
            for j in range(0, 4):
                start_row, start_col = widthPart * i, widthPart * (i+1) - 1
                end_row, end_col = heightPart * j, heightPart * (j+1) - 1
                crops.append(image[start_row:start_col, end_row:end_col])

        mosaic.append(cv2.hconcat([crops[5], crops[10], crops[12], crops[2]]))
        mosaic.append(cv2.hconcat([crops[7], crops[15], crops[0], crops[8]]))
        mosaic.append(cv2.hconcat([crops[11], crops[13], crops[1], crops[6]]))
        mosaic.append(cv2.hconcat([crops[3], crops[14], crops[9], crops[4]]))

        mosaicImage = cv2.vconcat(mosaic)

        return mosaicImage