class Dither:
    @staticmethod
    def applyHalftone(image, tech, changeDir):
        return image


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
                image[x][y] = newpixel
                error = float(pixel - newpixel * 255)

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