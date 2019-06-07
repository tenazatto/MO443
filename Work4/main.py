import argparse
import cv2
import os

import numpy as np

parser = argparse.ArgumentParser(description='MO443A Work 4')
parser.add_argument('-images', nargs='+', dest='images', required=True)
parser.add_argument('-detector', dest='detector', required=True)
parser.add_argument('-threshold', type=float, dest='threshold')
parser.add_argument('-show-keypoints', dest='show_keypoints', action='store_true')
parser.add_argument('-show-matches', dest='show_matches', action='store_true')

def apply_stitch(images, detector, threshold, showKeypoints, showMatches):
    if len(images) != 2:
        print('Invalid number of images! Put only 2 of them')
    else:
        imagePaths1 = images[0].split('/')
        imageName1 = imagePaths1[len(imagePaths1) - 1].split('.')[0]
        cvImage1 = cv2.imread(images[0])
        imagePaths2 = images[1].split('/')
        imageName2 = imagePaths2[len(imagePaths2) - 1].split('.')[0]
        cvImage2 = cv2.imread(images[1])
        imageResultName = imageName1 + '_' + imageName2 + '_' + detector + '.jpg'

        kp1, desc1 = detectKeypoints(images[0], detector, showKeypoints)
        kp2, desc2 = detectKeypoints(images[1], detector, showKeypoints)

        M = matchKeypoints(kp1, kp2, desc1, desc2, threshold)

        if M is None:
            return None

        resultW = cvImage1.shape[1] + cvImage2.shape[1]
        resultH = cvImage1.shape[0] if (cvImage1.shape[0] >= cvImage2.shape[0]) else cvImage2.shape[0]
        (matches, matches1to2, H, status) = M
        result = cv2.warpPerspective(cvImage1, H, (resultW, resultH))
        result[0:cvImage2.shape[0], 0:cvImage2.shape[1]] = cvImage2
        matchImg = cv2.drawMatches(cvImage1, kp1, cvImage2, kp2, matches1to2, None)

        if (showMatches):
            generateImage('./result_images/matches/', imageResultName, matchImg)

        result = postProcessing(result)
        generateImage('./result_images/stitch/', imageResultName, result)

def detectKeypoints(image, detector, showKeypoints):
    imagePaths = image.split('/')
    cvImage = cv2.imread(image)
    cvGray = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
    folderName = None
    kp, desc = None, None

    if (str.upper(detector) == 'SIFT'):
        folderName = './result_images/keypoints/sift/'
        sift = cv2.xfeatures2d.SIFT_create()
        kp, desc = sift.detectAndCompute(cvGray, None)
    elif (str.upper(detector) == 'SURF'):
        folderName = './result_images/keypoints/surf/'
        surf = cv2.xfeatures2d.SURF_create()
        kp, desc = surf.detectAndCompute(cvGray, None)
    elif (str.upper(detector) == 'BRIEF'):
        folderName = './result_images/keypoints/brief/'
        star = cv2.xfeatures2d.StarDetector_create()
        kp = star.detect(cvGray, None)
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        kp, desc = brief.compute(cvGray, kp)
    elif (str.upper(detector) == 'ORB'):
        folderName = './result_images/keypoints/orb/'
        orb = cv2.ORB_create()
        kp = orb.detect(cvGray, None)
        kp, desc = orb.compute(cvGray, kp)

    if (showKeypoints):
        generateImage(folderName, imagePaths[len(imagePaths) - 1], cv2.drawKeypoints(cvGray, kp, cvImage))

    return kp, desc


def matchKeypoints(kpsA, kpsB, featuresA, featuresB,
                   threshold, reprojThresh=4.0):
    threshold = 0.75 if threshold is None else threshold
    matcher = cv2.BFMatcher()
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []
    matches1to2 = []

    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * threshold:
            matches.append((m[0].trainIdx, m[0].queryIdx))
            matches1to2.append(m[0])

    if len(matches) > 4:
        ptsA = np.float32([kpsA[i].pt for (_, i) in matches])
        ptsB = np.float32([kpsB[i].pt for (i, _) in matches])

        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                         reprojThresh)
        if H is None:
            print('Homography Matrix not found')
            return None

        return (matches, matches1to2, H, status)

    print('Not enough points to stitch image')
    return None

def postProcessing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    img = img[y:y + h, x:x + w]

    return img

def generateImage(folderPath, fileName, image):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    cv2.imwrite(folderPath + fileName, image)

def main():
    args = parser.parse_args()

    apply_stitch(args.images, args.detector, args.threshold, args.show_keypoints, args.show_matches)


if __name__ == '__main__':
    main()