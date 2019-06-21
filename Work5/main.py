import argparse
import cv2
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

parser = argparse.ArgumentParser(description='MO443A Work 5')
parser.add_argument('-images', nargs='+', dest='images', required=True)
parser.add_argument('-ncolors', type=int, dest='ncolors', required=True)
parser.add_argument('-show-groups', dest='show_groups', action='store_true')

def applyImgQuantization(images, ncolors, show_groups):
    for image in images:
        imagePaths = image.split('/')
        imageNames = imagePaths[len(imagePaths) - 1].split('.')
        print('Generating ' + imageNames[0] + '.' + imageNames[1])
        cvImage = cv2.imread(image)
        cvImage = np.array(cvImage, dtype=np.float64) / 255

        w, h, d = original_shape = tuple(cvImage.shape)
        assert d == 3
        image_array = np.reshape(cvImage, (w * h, d))
        print('Number of colors in ' + imageNames[0] + ':', int(np.unique(image_array, axis=0).size / 3))

        max_samples = int(w * h * 0.01)
        image_array_sample = shuffle(image_array, random_state=0)[:max_samples]
        kmeans = KMeans(n_clusters=ncolors, random_state=0).fit(image_array_sample)

        labels = kmeans.predict(image_array)

        imgQuant = imageQuantization(kmeans.cluster_centers_, labels, w, h) * 255
        generateImage('./result_images/images/', imageNames[0] + str(ncolors) + '.' + imageNames[1], imgQuant)
        if (show_groups):
            generatePlot('./result_images/plots/', imageNames[0] + str(ncolors) + '.' + imageNames[1], image_array,
                         labels)

def imageQuantization(codebook, labels, w, h):
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

def generateImage(folderPath, fileName, image):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    cv2.imwrite(folderPath + fileName, image)

def generatePlot(folderPath, fileName, image_array, labels):
    fig = plt.figure(1, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(image_array[:, 0], image_array[:, 1], image_array[:, 2],
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title('KMeans ' + fileName)
    ax.dist = 12

    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    fig.savefig(folderPath + fileName)

def main():
    args = parser.parse_args()

    applyImgQuantization(args.images, args.ncolors, args.show_groups)


if __name__ == '__main__':
    main()