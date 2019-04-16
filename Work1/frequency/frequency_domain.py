import os
import numpy as np
import matplotlib.pyplot as plt


class FrequencyDomainFilter:
    @staticmethod
    def applyFilter(image, imageName, freq_type, freq_radius, freq_bandwidth, histogram):
        imageFreq = FrequencyDomainFilter.applyFrequencyDomain(image)
        magnitudeSpectrum = FrequencyDomainFilter.applyMagnitudeSpectrum(imageFreq)

        imageMask, maskedImage = FrequencyDomainFilter.maskImage(image, imageFreq, freq_type, freq_radius, freq_bandwidth)

        imageFiltered = FrequencyDomainFilter.applySpatialDomain(maskedImage)

        if (histogram):
            FrequencyDomainFilter.showHistogram(imageFiltered, freq_type, freq_radius, freq_bandwidth, imageName)

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
    def showHistogram(image, freq, radius, bandwidth, imageName):
        n, bins, patches = plt.hist(image.ravel(),256,[0,256])
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        if (freq > 2):
            plt.title('Histogram of ' + imageName + ' in filter ' + str(freq) + ', Radius ' + str(radius) +
                      ' and Bandwidth ' + str(bandwidth))
        else:
            plt.title('Histogram of ' + imageName + ' in filter ' + str(freq) + ' and Radius ' + str(radius))

        plt.text(23, 45, r'$\mu=15, b=3$')
        maxfreq = n.max()
        # Set a clean upper y-axis limit.
        plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        if not os.path.exists('./result_images/frequency/histogram'):
            os.makedirs('./result_images/frequency/histogram')
        plt.savefig('./result_images/frequency/histogram/histogram_filter'+ str(freq) + '_' + imageName)

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
        d = np.hypot(x, y)

        if (type == 1): #passa-baixa gaussiano
            exponent = -1*d**2/(2*d0**2)
            mask = np.exp(exponent)
        elif (type == 2): #passa-alta gaussiano
            exponent = -1*d**2/(2*d0**2)
            mask = 1 - np.exp(exponent)
        elif (type == 3): #passa-faixa gaussiano
            exponent = -1 * ((d ** 2 - d0 ** 2) / (w * d)) ** 2
            mask = np.exp(exponent)
        elif (type == 4):  # rejeita-faixa gaussiano
            exponent = -1 * ((d ** 2 - d0 ** 2) / (w * d)) ** 2
            mask = 1 - np.exp(exponent)

        return mask