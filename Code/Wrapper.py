#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Code starts here:

import numpy as np
import cv2
import torch
from Network.Network_supervised import HomographyModel as HomographyModelSuper
from Network.Network_unsupervised import HomographyModel as HomographyModelUnsuper

# Add any python libraries here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def DLT(H4Pt, corners):
    cornersB = corners + H4Pt.to("cpu").detach().numpy().view(-1, 4, 2)
    H = cv2.getPerspectiveTransform(corners, cornersB)
    return H

def getHomography(ModelPath, imgA, imgB):
    model = HomographyModelSuper()
    model.to(device)
    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint["model_state_dict"])

    # image to patch
    patchA, patchB, corners = 

    model.eval()
    with torch.no_grad():
        H4Pt = model(patchA, patchB)

    Homography = DLT(H4Pt, corners)

    return Homography

def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures
    ModelPath = "../Checkpoints_l2_square/29model.ckpt"

    """
    Read a set of images for Panorama stitching
    """


    """
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""


if __name__ == "__main__":
    main()
