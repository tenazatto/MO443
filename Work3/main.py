import argparse
import os

import cv2
import numpy as np

parser = argparse.ArgumentParser(description='MO443A Work 3')
parser.add_argument('-images', nargs='+', dest='images', required=True)
parser.add_argument('-word', dest='detect_word', action='store_true')
parser.add_argument('-count', dest='count_mode', action='store_true')

def main():
    args = parser.parse_args()

    if(args.detect_word):
        detectWord(args.images, args.count_mode)
    else:
        detectText(args.images, args.count_mode)

def detectText(images, count):
    imageFolder = './result_images/text/count/' if count else './result_images/text/detection/'

    for image in images:
        cvImage = cv2.imread(image, 0)

        invImage = cv2.bitwise_not(cvImage)

        generateImage(imageFolder, 'step0.pbm', invImage)

        print('Applying Step 1')
        strElStep1 = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
        cvImageStep1 = cv2.dilate(invImage, strElStep1)

        generateImage(imageFolder, 'step1.pbm', cvImageStep1)

        print('Applying Step 2')
        cvImageStep2 = cv2.erode(cvImageStep1, strElStep1)

        generateImage(imageFolder, 'step2.pbm', cvImageStep2)

        print('Applying Step 3')
        strElStep3 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 200))
        cvImageStep3 = cv2.dilate(invImage, strElStep3)

        generateImage(imageFolder, 'step3.pbm', cvImageStep3)

        print('Applying Step 4')
        cvImageStep4 = cv2.erode(cvImageStep3, strElStep3)

        generateImage(imageFolder, 'step4.pbm', cvImageStep4)

        print('Applying Step 5')
        cvImageStep5 = cv2.bitwise_and(cvImageStep2, cvImageStep4)

        generateImage(imageFolder, 'step5.pbm', cvImageStep5)

        print('Applying Step 6')
        strElStep6 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        cvImageStep6 = cv2.morphologyEx(cvImageStep5, cv2.MORPH_CLOSE, strElStep6)

        generateImage(imageFolder, 'step6.pbm', cvImageStep6)

        cvImageStep6 = cv2.bitwise_not(cvImageStep6)
        generateImage(imageFolder, 'step6-2.pbm', cvImageStep6)

        print('Applying Step 7')
        contours, hierarchy = cv2.findContours(cvImageStep6, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cvImageStep7 = cvImageStep6
        cvImageStep9 = cvImage

        print('Applying Step 8')
        i = 1
        text = 0
        contoursFiltered = contourFilter(contours, 100, 10)
        textContours = []
        for c in contoursFiltered:
            x, y, w, h = c[0], c[1], c[2], c[3]
            ratioBlackPixels, ratioTransitions = getStats(cvImageStep7, x, y, w, h, c, i)
            cv2.rectangle(cvImageStep7, (x, y), (x + w, y + h), (0, 0, 0), 1)
            cv2.rectangle(cvImageStep9, (x, y), (x + w, y + h), (0, 0, 0), 1)
            print('Applying Step 9')
            txt = str(i)
            if (ratioBlackPixels > 0.3) and (ratioBlackPixels < 0.9) \
                    and (ratioTransitions > 1.86) and (ratioTransitions < 1.97):
                print('Text detected')
                text += 1
                textContours.append(c)
                if count:
                    txt = str(text)
                    putTextOnImage(cvImageStep9, txt, text, x, y, w, h, 20)
                else:
                    txt = str(i) + ' - Text'
                    putTextOnImage(cvImageStep9, txt, i, x, y, w, h, 100)
            else:
                print('Not Text detected')
                if not count:
                    txt += ' - Not Text'
                    putTextOnImage(cvImageStep9, txt, i, x, y, w, h, 100)

            i += 1

        if count:
            lines = text
            imageSect = int(len(cvImage) / 2)
            for i in range(0, len(textContours) - 1):
                for j in range(i + 1, len(textContours)):
                    # Don't validate texts from different sections
                    if (textContours[i][0] <= imageSect and textContours[j][0] > imageSect):
                        continue
                    if (textContours[i][0] > imageSect and textContours[j][0] <= imageSect):
                        continue
                    # Remove texts in same line from line count
                    if (textContours[i][1] - textContours[j][1] < 10):
                        lines -= 1

            for i in range(0, len(textContours)):
                # Remove header texts from line count
                if (textContours[i][1] < 50):
                    lines -= 1

            print('Number of lines:', lines)

        generateImage(imageFolder, 'step7.pbm', cvImageStep7)
        generateImage(imageFolder, 'step9.pbm', cvImageStep9)

def detectWord(images, count):
    imageFolder = './result_images/word/count/' if count else './result_images/word/detection/'

    for image in images:
        cvImage = cv2.imread(image, 0)

        invImage = cv2.bitwise_not(cvImage)

        generateImage(imageFolder, 'step0.pbm', invImage)

        print('Applying Step 1')
        strElStep1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        cvImageStep1 = cv2.dilate(invImage, strElStep1)

        generateImage(imageFolder, 'step1.pbm', cvImageStep1)

        print('Applying Step 2')
        cvImageStep2 = cv2.erode(cvImageStep1, strElStep1)

        generateImage(imageFolder, 'step2.pbm', cvImageStep2)

        print('Applying Step 3')
        strElStep3 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60))
        cvImageStep3 = cv2.dilate(invImage, strElStep3)

        generateImage(imageFolder, 'step3.pbm', cvImageStep3)

        print('Applying Step 4')
        cvImageStep4 = cv2.erode(cvImageStep3, strElStep3)

        generateImage(imageFolder, 'step4.pbm', cvImageStep4)

        print('Applying Step 5')
        cvImageStep5 = cv2.bitwise_and(cvImageStep2, cvImageStep4)

        generateImage(imageFolder, 'step5.pbm', cvImageStep5)

        print('Applying Step 6')
        strElStep6 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        cvImageStep6 = cv2.morphologyEx(cvImageStep5, cv2.MORPH_CLOSE, strElStep6)

        generateImage(imageFolder, 'step6.pbm', cvImageStep6)

        cvImageStep6 = cv2.bitwise_not(cvImageStep6)
        generateImage(imageFolder, 'step6-2.pbm', cvImageStep6)

        print('Applying Step 7')
        contours, hierarchy = cv2.findContours(cvImageStep6, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cvImageStep7 = cvImageStep6
        cvImageStep9 = cvImage

        print('Applying Step 8')
        i = 1
        word = 0
        contoursFiltered = contourFilter(contours, 300, 10)
        for c in contoursFiltered:
            x, y, w, h = c[0], c[1], c[2], c[3]
            ratioBlackPixels, ratioTransitions = getStats(cvImageStep7, x, y, w, h, c, i)
            cv2.rectangle(cvImageStep7, (x, y), (x + w, y + h), (0, 0, 0), 1)
            cv2.rectangle(cvImageStep9, (x, y), (x + w, y + h), (0, 0, 0), 1)
            print('Applying Step 9')

            if (ratioBlackPixels > 0.23) and (ratioBlackPixels < 0.9) \
                    and (ratioTransitions > 1.61) and (ratioTransitions < 2):
                print('Word detected')
                word += 1
                if count:
                    txt = str(word)
                    putTextOnImage(cvImageStep9, txt, word, x, y, w, h, 20)
                else:
                    txt = 'W' + str(i)
                    putTextOnImage(cvImageStep9, txt, i, x, y, w, h, 20)
            else:
                print('Not Word detected')
                if not count:
                    txt = 'NW' + str(i)
                    putTextOnImage(cvImageStep9, txt, i, x, y, w, h, 20)

            i += 1

        generateImage(imageFolder, 'step7.pbm', cvImageStep7)
        generateImage(imageFolder, 'step9.pbm', cvImageStep9)
        print('Number of words:', word)

def generateImage(folderPath, fileName, image):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    cv2.imwrite(folderPath + fileName, image)

def contourFilter(contours, minPixels, minHeight):
    contoursFiltered = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if not np.all(c == contours[0]) | (w * h < minPixels) | (h < minHeight):
            contoursFiltered.append((x, y, w, h))

    return contoursFiltered

def getStats(image, x, y, w, h, c, i):
    numPixels = w * h
    numBlackPixels = 0
    horTransitionsWhiteToBlack = 0
    verTransitionsWhiteToBlack = 0

    for imgx in range(x-1, x+w-1):
        for imgy in range(y-1, y + h-1):
            if (image[imgy][imgx] == 0):
                numBlackPixels += 1
            if (image[imgy][imgx] == 255 & image[imgy][imgx+1] == 0 & imgx < x+w-1):
                horTransitionsWhiteToBlack += 1
            if (image[imgy][imgx] == 255 & image[imgy+1][imgx] == 0& imgy < y + h-1):
                verTransitionsWhiteToBlack += 1
            if (image[imgy][imgx] == 0 & image[imgy + 1][imgx] == 255 & imgx < x+w-1):
                horTransitionsWhiteToBlack += 1
            if (image[imgy][imgx] == 0 & image[imgy][imgx + 1] == 255 & imgx < x+w-1):
                verTransitionsWhiteToBlack += 1

    numTransitionsWhiteToBlack = horTransitionsWhiteToBlack + verTransitionsWhiteToBlack
    ratioBlackPixels = numBlackPixels / numPixels
    ratioTransitions = numTransitionsWhiteToBlack / numBlackPixels

    print('Stats of contour ', i)
    print('Width and Height pixels: ', w, h)
    print('Total pixels: ', numPixels)
    print('Total black pixels: ', numBlackPixels)
    print('Ratio black pixels: ', ratioBlackPixels)
    print('Horizontal Transitions White To Black: ', horTransitionsWhiteToBlack)
    print('Vertical Transitions White To Black: ', verTransitionsWhiteToBlack)
    print('Total Transitions White To Black: ', numTransitionsWhiteToBlack)
    print('Ratio Transitions White To Black: ', ratioTransitions)

    return ratioBlackPixels, ratioTransitions

def putTextOnImage(image, text, counter, x, y, w, h, adjust):
    if counter < 10:
        cv2.putText(image, text, (x + w - adjust, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
    elif counter < 100:
        cv2.putText(image, text, (x + w - 2 * adjust, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
    else:
        cv2.putText(image, text, (x + w - 3 * adjust, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

if __name__ == '__main__':
    main()