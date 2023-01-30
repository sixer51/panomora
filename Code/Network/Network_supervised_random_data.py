"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
# import kornia  # You can use this to get the transform and warp in this project
import math

# Don't generate pyc codes
sys.dont_write_bytecode = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def LossFn(homography, predHomography):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################
    # loss = 0
    # for i in range(len(homography)):
    #     loss += (homography[i] - predHomography[i]) ** 2
    # print(loss)
    # print(predHomography, homography)
    # print(torch.sub(predHomography, homography))
    # print(torch.linalg.norm(torch.sub(predHomography, homography), dim = 1).size())

    # return torch.sum(torch.linalg.norm(torch.sub(predHomography, homography), dim = 1))
    # return torch.sum(torch.linalg.norm(torch.sub(predHomography, homography), dim = 1))
    return torch.sum(torch.linalg.norm(torch.sub(predHomography, homography) ** 2, dim = 1))


class HomographyModel(nn.Module):
    def __init__(self):
    # def __init__(self, hparams):
        super(HomographyModel, self).__init__()
        # self.hparams = hparams
        self.model = Net()

    def forward(self, a, b):
        return self.model(a, b)

    def training_step(self, batch):
        patchA, patchB, homography = batch
        predHomography = self.model(patchA, patchB) # get predicted homography
        loss = LossFn(homography, predHomography)
        logs = {"loss": loss}
        # return {"loss": loss, "log": logs}
        return loss

    def validation_step(self, batch):
        patchA, patchB, homography = batch
        predHomography = self.model(patchA, patchB) # get predicted homography
        loss = LossFn(homography, predHomography)
        # img_a, patch_a, patch_b, corners, gt = batch
        # delta = self.model(patch_a, patch_b)
        # loss = LossFn(delta, img_a, patch_b, corners)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}

    def epoch_end(self, epoch, iteration, result):
        print(result)
        loss = result["val_loss"].item()
        avg_loss = loss / 64
        # avg_loss = torch.stack([x["val_loss"] for x in result]).mean()
        print("Epoch [{}], iteration [{}], loss: {:.4f}, avg_loss: {:.4f}, ".format(epoch, iteration, loss, avg_loss))



class Net(nn.Module):
    def __init__(self):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()
        #############################
        # Fill your network initialization of choice here!
        #############################
        self.network = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Flatten(), 
            nn.Linear(128*16*16, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 8))

        # self.network = nn.Sequential(
        #     nn.Conv2d(6, 128, 3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Dropout2d(0.5),
        #     nn.Conv2d(128, 128, 3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2),

        #     nn.Conv2d(128, 128, 3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, 3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2),

        #     nn.Conv2d(128, 256, 3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2),

        #     nn.Conv2d(256, 256, 3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),

        #     nn.Flatten(), 
        #     nn.Linear(256*16*16, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     # nn.Linear(4096, 1024),
        #     # nn.ReLU(),
        #     nn.Linear(1024, 8))

    def forward(self, xa, xb):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################

        #stack image a and b
        patchA = np.float32(xa)
        patchA = np.swapaxes(patchA, 1, 3)
        patchB = np.float32(xb)
        patchB = np.swapaxes(patchB, 1, 3)

        x = np.hstack((patchA, patchB))
        x = torch.from_numpy(x).to(device)

        out = self.network(x)

        return out
