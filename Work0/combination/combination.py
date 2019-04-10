class ImageCombination:
    @staticmethod
    def combineImages(combineValues, image1, image2):
        return (combineValues[0] * image1 + combineValues[1] * image2) / (combineValues[0] + combineValues[1])