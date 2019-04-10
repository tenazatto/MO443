import numpy as np


class FrequencyDomainFilter:
    @staticmethod
    def applyFilter(image, freq_type, freq_radius, freq_bandwidth):
        imageFreq = FrequencyDomainFilter.applyFrequencyDomain(image)
        magnitudeSpectrum = FrequencyDomainFilter.applyMagnitudeSpectrum(imageFreq)

        imageMask, maskedImage = FrequencyDomainFilter.maskImage(image, imageFreq, freq_type, freq_radius, freq_bandwidth)

        imageFiltered = FrequencyDomainFilter.applySpatialDomain(maskedImage)

        return imageMask * 255, maskedImage, imageFiltered, magnitudeSpectrum;

    @staticmethod
    def applyFrequencyDomain(image):
        h = np.fft.fft2(image)
        hshift = np.fft.fftshift(h)

        return hshift

    @staticmethod
    def maskImage(image, image_freq, type, d0, w):
        imageMask = FrequencyDomainFilter.getMask(image, image_freq, type, d0, w)
        maskedImage = image_freq * imageMask

        return imageMask, maskedImage

    @staticmethod
    def applySpatialDomain(masked_image):
        inverseHshift = np.fft.ifftshift(masked_image)
        imageFiltered = np.fft.ifft2(inverseHshift)
        imageFiltered = np.abs(imageFiltered)

        return imageFiltered

    @staticmethod
    def applyMagnitudeSpectrum(masked_image):
        return 20 * np.log(np.abs(masked_image))

    @staticmethod
    def getMask(image, image_freq, type, d0, w):
        rows = image.shape[0]
        cols = image.shape[1]
        crow, ccol = rows / 2, cols / 2

        n = len(image_freq)
        y, x = np.ogrid[-crow:n - crow, -ccol:n - ccol]
        d = np.sqrt(x*x + y*y)

        if (type == 1): #passa-baixa gaussiano
            exponent = -1*d**2/(2*d0**2)
            mask = np.exp(exponent)
        elif (type == 2): #passa-alta gaussiano
            exponent = -1*d**2/(2*d0**2)
            mask = 1 - np.exp(exponent)
        elif (type == 3): #passa-faixa gaussiano
            exponent = -1*((d**2 - d0**2)/(w*d))**2
            mask = np.exp(exponent)
        elif (type == 4):  # rejeita-faixa gaussiano
            exponent = -1 * ((d ** 2 - d0 ** 2) / (w * d)) ** 2
            mask = 1 - np.exp(exponent)

        return mask