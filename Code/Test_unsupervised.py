#!/usr/bin/env python
"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from torchvision.transforms import ToTensor
import argparse
from Network.Network_unsupervised import HomographyModel
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
from DataGeneration import dataGeneration


# Don't generate pyc codes
sys.dont_write_bytecode = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def SetupAll():
    """
    Outputs:
    ImageSize - Size of the Image
    """
    # Image Input Shape
    ImageSize = [128, 128, 6]

    return ImageSize


def StandardizeInputs(Img):
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################
    return Img


def ReadImages(Img):
    """
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """
    I1 = Img

    if I1 is None:
        # OpenCV returns empty list if image is not read!
        print("ERROR: Image I1 cannot be read")
        sys.exit()

    I1S = StandardizeInputs(np.float32(I1))

    I1Combined = np.expand_dims(I1S, axis=0)

    return I1Combined, I1


def TestOperation(ModelPath):
    """
    Inputs:
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    TestSet - The test dataset
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to /content/data/TxtFiles/PredOut.txt
    """
    # Predict output with forward pass, MiniBatchSize for Test is 1
    model = HomographyModel()
    model.to(device)

    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint["model_state_dict"])
    print(
        "Number of parameters in this model are %d " % len(model.state_dict().items())
    )

    # OutSaveT = open(LabelsPathPred, "w")
    EPE_all = []
    for count in tqdm(range(30)):

        # read image
        RandImageName = "../Data/Val/{}.jpg".format(count + 1)

        patchA, patchB, H4Pts, cornersA, _, _ = dataGeneration(RandImageName)
        # patchs = np.dstack((patchA, patchB))
        # patchs = torch.from_numpy(np.expand_dims(patchs, axis=0)).to(device)
        patchA = torch.from_numpy(np.expand_dims(patchA, axis=0)).float().to(device)
        patchB = torch.from_numpy(np.expand_dims(patchB, axis=0)).float().to(device)
        H4Pts = torch.tensor(H4Pts)

        model.eval()
        with torch.no_grad():
            Pred = model(patchA, patchB)
            # Pred = model(patchs)
        EPE = torch.sum(torch.linalg.norm(torch.sub(Pred.to("cpu"), H4Pts), dim = 1)).item()
        # accuracy = Accuracy(Pred.to("cpu").detach().numpy(), Coordinates.detach().numpy())
        print("idx: {}, error = {}".format(count, EPE))
        EPE_all.append(EPE)

        img_w_corner = cv2.imread(RandImageName)
        cornersB = cornersA + np.reshape(H4Pts.numpy(), (4,2))
        PredCornersB = cornersA + np.reshape(Pred.to("cpu").detach().numpy(), (4,2))
        print(Pred.to("cpu").detach().numpy())

        isClosed = True
        thickness = 2
        img_w_corner = cv2.polylines(img_w_corner, [cornersA.astype(int)], isClosed, (255, 0, 0), thickness)
        img_w_corner = cv2.polylines(img_w_corner, [cornersB.astype(int)], isClosed, (0, 0, 255), thickness)
        img_w_corner = cv2.polylines(img_w_corner, [PredCornersB.astype(int)], isClosed, (0, 255, 0), thickness)
        cv2.imwrite("../Data/val_result/{}.jpg".format(count + 1), img_w_corner)

        # OutSaveT.write(str(PredT) + "\n")
    # OutSaveT.close()

    EPE_aver = sum(EPE_all) / len(EPE_all)
    print("average EPE: {}".format(EPE_aver))


def ReadLabels(LabelsPathTest, LabelsPathPred):
    if not (os.path.isfile(LabelsPathTest)):
        print("ERROR: Test Labels do not exist in " + LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, "r")
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if not (os.path.isfile(LabelsPathPred)):
        print("ERROR: Pred Labels do not exist in " + LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, "r")
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())

    return LabelTest, LabelPred


def main():
    """
    Inputs:
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--ModelPath",
        dest="ModelPath",
        default="../Checkpoints_unsupervised_dropout/49model.ckpt",
        # default="../Checkpoints_l2_square/26model.ckpt",
        help="Path to load latest model from, Default:ModelPath",
    )
    Parser.add_argument(
        "--BasePath",
        dest="BasePath",
        default="../Data/Val/",
        help="Path to load images from, Default:BasePath",
    )
    Parser.add_argument(
        "--LabelsPath",
        dest="LabelsPath",
        default="./TxtFiles/LabelsTest.txt",
        help="Path of labels file, Default:./TxtFiles/LabelsTest.txt",
    )
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    LabelsPath = Args.LabelsPath

    # Setup all needed parameters including file reading
    # ImageSize, DataPath = SetupAll(BasePath)

    # Define PlaceHolder variables for Input and Predicted output
    # ImgPH = tf.placeholder("float", shape=(1, ImageSize[0], ImageSize[1], 3))
    # LabelsPathPred = "./TxtFiles/PredOut.txt"  # Path to save predicted labels

    TestOperation(ModelPath)


if __name__ == "__main__":
    main()
