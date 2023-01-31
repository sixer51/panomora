#!/usr/bin/evn python

import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import copy

def dataGeneration(imgPath, patchSize = 128, pertubation = 32):
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imgHeight, imgWidth = img.shape
    # imgHeight, imgWidth, channel = img.shape
    maxWidth = imgWidth - patchSize
    maxHeight = imgHeight - patchSize
    widthOrigin = random.randint(0, maxWidth)
    heightOrigin = random.randint(0, maxHeight)
    patch = img[heightOrigin: heightOrigin + patchSize,
                widthOrigin: widthOrigin + patchSize]

    # add pertubation
    topLeft = (widthOrigin, heightOrigin)
    topRight = (widthOrigin + patchSize, heightOrigin)
    bottomLeft = (widthOrigin, heightOrigin + patchSize)
    bottomRight = (widthOrigin + patchSize, heightOrigin + patchSize)
    corners = [topLeft, topRight, bottomRight, bottomLeft]
    imgCorner = copy.copy(img)

    pertubratedCorners = []
    for corner in corners:
        x = min(imgWidth, max(0, corner[0] + random.randint(-pertubation, pertubation)))
        y = min(imgHeight, max(0, corner[1] + random.randint(-pertubation, pertubation)))
        pertubratedCorner = (x, y)
        pertubratedCorners.append(pertubratedCorner)
        imgCorner = cv2.circle(imgCorner, corner, radius=3, color=(255, 0, 0), thickness=-1)
        imgCorner = cv2.circle(imgCorner, pertubratedCorner, radius=3, color=(0, 0, 255), thickness=-1)
    corners = np.float32(corners)
    pertubratedCorners = np.float32(pertubratedCorners)

    M = cv2.getPerspectiveTransform(corners, pertubratedCorners)
    Minv = np.linalg.inv(M)
    imgTransformed = cv2.warpPerspective(img, Minv, (imgWidth, imgHeight))
    imgTransformedCorner = cv2.warpPerspective(imgCorner, Minv, (imgWidth, imgHeight))
    transformedPatch = imgTransformed[heightOrigin: heightOrigin + patchSize,
                                      widthOrigin: widthOrigin + patchSize]

    H4Pt = pertubratedCorners - corners
    H4Pt = np.reshape(H4Pt, 8)

    return patch, transformedPatch, H4Pt, np.array(corners)

# labelWriter = open("./Code/TxtFiles/LabelsTrainSupervised.txt", 'a')
# for i in range(1, 5001):
#     imgPath = "../Data/Train/{}.jpg".format(i)
#     patchSize = 128
#     pertubation = 32
#     patch, transformedPatch, H4Pt, _ = dataGeneration(imgPath, patchSize, pertubation)

#     H4PtStr = np.array2string(H4Pt, separator=',')[1: -1] + '\n'
#     labelWriter.writelines(H4PtStr)
#     cv2.imwrite("../Data/data_patch_train/patchA_{}.jpg".format(i), patch)
#     cv2.imwrite("../Data/data_patch_train/patchB_{}.jpg".format(i), transformedPatch)