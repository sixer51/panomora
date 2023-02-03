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
# termcolor, do (pip install termcolor)

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
from Network.Network_unsupervised import HomographyModel, weight_init
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
from torchvision import transforms
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from DataGeneration import dataGeneration
import visdom

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layoutopts = {'plotly': {'yaxis': {'type': 'log','autorange': True}}}
vis = visdom.Visdom()
loss_window = vis.line(
    Y=torch.zeros((1)).cpu(),
    X=torch.zeros((1)).cpu(),
    opts=dict(xlabel='Iteration',ylabel='Loss',title='training loss',legend=['Loss'],layoutopts = layoutopts))

# val_loss_window = vis.line(
#     Y=torch.zeros((1)).cpu(),
#     X=torch.zeros((1)).cpu(),
#     opts=dict(xlabel='Iteration',ylabel='Loss',title='validation loss',legend=['Loss'],layoutopts = layoutopts))

# transform_norm = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean, std)
# ])

def GenerateBatch(BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates - Coordinatess corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    CoordinatesBatch - Batch of coordinates
    """
    PatchABatch = []
    PatchBBatch = []
    PatchBatch = []
    CoordinatesBatch = []
    cornersBatch = []
    imgBatch = []
    HomoBatch = []

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        # RandIdx = random.randint(1, 49)
        RandIdx = random.randint(0, len(DirNamesTrain) - 1)
        # print(len(DirNamesTrain), len(TrainCoordinates))

        RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + ".jpg"
        patchA, patchB, Coordinates, corners, img, H = dataGeneration(RandImageName)

        # RandPatchA = BasePath + "/data_patch_train/patchA_"+ str(RandIdx) + ".jpg"
        # RandPatchB = BasePath + "/data_patch_train/patchB_"+ str(RandIdx) + ".jpg"
        ImageNum += 1

        ##########################################################
        # Add any standardization or data augmentation here!
        ##########################################################
        # patchA = np.float32(cv2.imread(RandPatchA))
        # patchB = np.float32(cv2.imread(RandPatchB))
        # Coordinates = TrainCoordinates[RandIdx - 1]

        # Append All Images and Mask
        patchA = torch.from_numpy(np.float32(patchA))
        # patchA_mean = patchA.mean(dim=[0,1])
        # patchA_std = patchA.std(dim=[0,1])
        # patchA_transform = transforms.Compose([
        #                 transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
        #             ])
        # # patchA_transform = transforms.Normalize(patchA_mean,patchA_std)
        # patchA = patchA.view(3,128,128)
        # patchA = patchA_transform(patchA)
        # # patchA = patchA.view(128,128,3)

        patchB = torch.from_numpy(np.float32(patchB))
        # patchB_mean = patchB.mean(dim=[0,1])
        # patchB_std = patchB.std(dim=[0,1])
        # patchB_transform = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
        # patchB = patchB.view(3,128,128)
        # patchB = patchB_transform(patchB)
        # # patchB = patchB.view(128,128,3)

        PatchABatch.append(patchA)
        PatchBBatch.append(patchB)

        # img = torch.from_numpy(img)
        imgBatch.append(img)
        HomoBatch.append(torch.from_numpy(H))
        # patchs = torch.from_numpy(np.dstack((patchA, patchB)))
        # mean, std= torch.mean(patchs), torch.std(patchs)
        # patchs = (patchs - mean.float() + 1e-7) / std.float()
        # PatchBatch.append(patchs)

        Coordinates = torch.tensor(Coordinates)
        # mean, std= torch.mean(Coordinates), torch.std(Coordinates)
        # Coordinates = (Coordinates - mean.float() + 1e-7) / std.float()
        CoordinatesBatch.append(Coordinates)

        corners = torch.from_numpy(corners)
        cornersBatch.append(corners)

    return torch.stack(PatchABatch).to(device), torch.stack(PatchBBatch).to(device), torch.stack(CoordinatesBatch).to(device), torch.stack(cornersBatch).to(device), imgBatch, HomoBatch
    # return torch.stack(PatchBatch).to(device), torch.stack(CoordinatesBatch).to(device) , torch.stack(cornersBatch).to(device)


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def TrainOperation(
    DirNamesTrain,
    TrainCoordinates,
    NumTrainSamples,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    ModelType,
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    model = HomographyModel()
    model.apply(weight_init)
    model.to(device)

    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    # Optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    Optimizer = torch.optim.SGD(model.parameters(), lr=0.0005,momentum=0.9)
    # Optimizer = AdamW(model.parameters(), lr = 0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(Optimizer, milestones=[15, 25], gamma=0.1)

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    # if LatestFile is not None:
    #     CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
    #     # Extract only numbers from the name
    #     StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
    #     model.load_state_dict(CheckPoint["model_state_dict"])
    #     print("Loaded latest checkpoint with the name " + LatestFile + "....")
    # else:
    StartEpoch = 0
    print("New model initialized....")

    # CheckPoint = torch.load("../Checkpoints_l2_square/2model.ckpt")
    # StartEpoch = 0
    # model.load_state_dict(CheckPoint["model_state_dict"])
    # print("Loaded a checkpoint")

    loss_all = []
    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        loss_this_epoch = 0
        # Batch = GenerateBatch(
        #     BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize
        # )
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            # I1Batch, CoordinatesBatch = GenerateBatch(
            Batch = GenerateBatch(
                BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize
            )

            # Predict output with forward pass
            # torch.cuda.empty_cache()
            print("training")
            model.train()
            # torch.set_grad_enabled(True)

            Optimizer.zero_grad()
            # patchs, homography = Batch
            # LossThisBatch = model(patchs)
            LossThisBatch = model.training_step(Batch, Epochs*NumIterationsPerEpoch + PerEpochCounter)
            # PredicatedCoordinatesBatch = model(I1Batch)
            # LossThisBatch = LossFn(PredicatedCoordinatesBatch, CoordinatesBatch)

            # Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            # print("validating")
            # model.eval()
            # with torch.no_grad():
            #     result = model.validation_step(Batch)

            # model.iteration_end(Epochs + 1, Epochs*NumIterationsPerEpoch + PerEpochCounter, MiniBatchSize, result)
            # loss_this_epoch += result["val_loss"]
            vis.line(X=torch.ones((1,1)).cpu()*Epochs*NumIterationsPerEpoch + PerEpochCounter,Y=torch.Tensor([LossThisBatch]).unsqueeze(0).cpu(),win=loss_window,update='append')
            # vis.line(X=torch.ones((1,1)).cpu()*Epochs*NumIterationsPerEpoch + PerEpochCounter,Y=torch.Tensor([LossThisBatch]).unsqueeze(0).cpu() / MiniBatchSize,win=loss_window,update='append')
            # vis.line(X=torch.ones((1,1)).cpu()*Epochs*NumIterationsPerEpoch + PerEpochCounter,Y=torch.Tensor([result["val_loss"]]).unsqueeze(0).cpu()  / MiniBatchSize,win=val_loss_window,update='append')
            
            # Tensorboard
            # Writer.add_scalar(
            #     "LossEveryIter",
            #     result["val_loss"],
            #     Epochs * NumIterationsPerEpoch + PerEpochCounter,
            # )
            # # If you don't flush the tensorboard doesn't update until a lot of iterations!
            # Writer.flush()

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": LossThisBatch,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")

        # Update loss_all
        # loss_all.append(loss_this_epoch.item() / (MiniBatchSize*NumIterationsPerEpoch))
        
        scheduler.step()

    # print(loss_all)
    # epochs = np.arange(1, NumEpochs + 1)
    # plt.plot(epochs, loss_all)
    # plt.ylabel("Train loss")
    # plt.xlabel("Epochs")
    # plt.show()


def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="../Data",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="../Checkpoints_unsupervised/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Unsup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=30,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=64,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    (
        DirNamesTrain,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainCoordinates,
        NumClasses,
    ) = SetupAll(BasePath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(
        DirNamesTrain,
        TrainCoordinates,
        NumTrainSamples,
        ImageSize,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        BasePath,
        LogsPath,
        ModelType,
    )


if __name__ == "__main__":
    main()
