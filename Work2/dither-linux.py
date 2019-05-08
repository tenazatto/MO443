import numpy as np


class Dither:
    @staticmethod
    def applyHalftone(image, tech, changeDir):
        (iH, iW) = image.shape[:2]
        backward = False
        halftoneImage = None
        iteration = range(0, 0)
        if tech == 1:
            halftoneImage = np.zeros((iH * 3, iW * 3))
        else:
            halftoneImage = np.zeros((iH * 4, iW * 4))

        for y in range(0, iW):
            if backward:
                iteration = range(iH, 0)
            else:
                iteration = range(0, iH)

            for x in iteration:
                halftoneImage = Dither.orderedDither(halftoneImage, tech, image[x][y], x, y, backward)

            if(changeDir):
                backward = not(backward)

        return halftoneImage

    @staticmethod
    def orderedDither(halftoneImage, tech, value, x, y, backward):
        if tech == 1:
            valueNormalized = int(value / 255 * 9)
            dither = np.zeros((3, 3))
            if (backward):
                dither[0][2] = 0 if valueNormalized < 7 else 255
                dither[0][1] = 0 if valueNormalized < 9 else 255
                dither[0][0] = 0 if valueNormalized < 5 else 255
                dither[1][2] = 0 if valueNormalized < 2 else 255
                dither[1][1] = 0 if valueNormalized < 1 else 255
                dither[1][0] = 0 if valueNormalized < 4 else 255
                dither[2][2] = 0 if valueNormalized < 6 else 255
                dither[2][1] = 0 if valueNormalized < 3 else 255
                dither[2][0] = 0 if valueNormalized < 8 else 255
            else:
                dither[0][0] = 0 if valueNormalized < 7 else 255
                dither[0][1] = 0 if valueNormalized < 9 else 255
                dither[0][2] = 0 if valueNormalized < 5 else 255
                dither[1][0] = 0 if valueNormalized < 2 else 255
                dither[1][1] = 0 if valueNormalized < 1 else 255
                dither[1][2] = 0 if valueNormalized < 4 else 255
                dither[2][0] = 0 if valueNormalized < 6 else 255
                dither[2][1] = 0 if valueNormalized < 3 else 255
                dither[2][2] = 0 if valueNormalized < 8 else 255

            for hy in range(3 * y, 3 * y + 2):
                for hx in range(3 * x, 3 * x + 2):
                    halftoneImage[hx][hy] = dither[hx - 3 * x][hy - 3 * y]
        else:
            valueNormalized = int(value / 255 * 16)
            dither = np.zeros((4, 4))
            if (backward):
                dither[0][3] = 0 if valueNormalized < 1 else 255
                dither[0][2] = 0 if valueNormalized < 13 else 255
                dither[0][1] = 0 if valueNormalized < 4 else 255
                dither[0][0] = 0 if valueNormalized < 16 else 255
                dither[1][3] = 0 if valueNormalized < 9 else 255
                dither[1][2] = 0 if valueNormalized < 5 else 255
                dither[1][1] = 0 if valueNormalized < 12 else 255
                dither[1][0] = 0 if valueNormalized < 8 else 255
                dither[2][3] = 0 if valueNormalized < 3 else 255
                dither[2][2] = 0 if valueNormalized < 15 else 255
                dither[2][1] = 0 if valueNormalized < 2 else 255
                dither[2][0] = 0 if valueNormalized < 14 else 255
                dither[3][3] = 0 if valueNormalized < 11 else 255
                dither[3][2] = 0 if valueNormalized < 7 else 255
                dither[3][1] = 0 if valueNormalized < 10 else 255
                dither[3][0] = 0 if valueNormalized < 6 else 255
            else:
                dither[0][0] = 0 if valueNormalized < 1 else 255
                dither[0][1] = 0 if valueNormalized < 13 else 255
                dither[0][2] = 0 if valueNormalized < 4 else 255
                dither[0][3] = 0 if valueNormalized < 16 else 255
                dither[1][0] = 0 if valueNormalized < 9 else 255
                dither[1][1] = 0 if valueNormalized < 5 else 255
                dither[1][2] = 0 if valueNormalized < 12 else 255
                dither[1][3] = 0 if valueNormalized < 8 else 255
                dither[2][0] = 0 if valueNormalized < 3 else 255
                dither[2][1] = 0 if valueNormalized < 15 else 255
                dither[2][2] = 0 if valueNormalized < 2 else 255
                dither[2][3] = 0 if valueNormalized < 14 else 255
                dither[3][0] = 0 if valueNormalized < 11 else 255
                dither[3][1] = 0 if valueNormalized < 7 else 255
                dither[3][2] = 0 if valueNormalized < 10 else 255
                dither[3][3] = 0 if valueNormalized < 6 else 255

            for hy in range(4 * y, 4 * y + 3):
                for hx in range(4 * x, 4 * x + 3):
                    halftoneImage[hx][hy] = dither[hx - 4 * x][hy - 4 * y]

        return halftoneImage

    @staticmethod
    def applyColorBounds(v):
        if v > 255:
            v = 255
        if v < 0:
            v = 0
        return v


    @staticmethod
    def applyFloydSteinberg(image, changeDir):
        (iH, iW) = image.shape[:2]
        backward = False
        iteration = range(0, 0)

        for y in range(0, iW):
            if backward:
                iteration = range(iH, 0)
            else:
                iteration = range(0, iH)

            for x in iteration:
                pixel = image[x][y]
                newpixel = int(round(pixel / 255))
                image[x][y] = newpixel * 255
                error = float(pixel - newpixel * 255)

                if (backward):
                    if (x > 0):
                        image[x-1][y] = Dither.applyColorBounds(image[x-1][y] + (7/16 * error))
                    if (x < iH - 1) and (y < iW - 1):
                        image[x+1][y+1] = Dither.applyColorBounds(image[x+1][y+1] + (3/16 * error))
                    if (y < iW - 1):
                        image[x][y+1] = Dither.applyColorBounds(image[x][y+1] + (5/16 * error))
                    if (x > 0) and (y < iW - 1):
                        image[x-1][y+1] = Dither.applyColorBounds(image[x-1][y+1] + (1/16 * error))
                else:
                    if (x < iH - 1):
                        image[x+1][y] = Dither.applyColorBounds(image[x+1][y] + (7/16 * error))
                    if (x > 0) and (y < iW - 1):
                        image[x-1][y+1] = Dither.applyColorBounds(image[x-1][y+1] + (3/16 * error))
                    if (y < iW - 1):
                        image[x][y+1] = Dither.applyColorBounds(image[x][y+1] + (5/16 * error))
                    if (x < iH - 1) and (y < iW - 1):
                        image[x+1][y+1] = Dither.applyColorBounds(image[x+1][y+1] + (1/16 * error))

            if(changeDir):
                backward = not(backward)


        return image
