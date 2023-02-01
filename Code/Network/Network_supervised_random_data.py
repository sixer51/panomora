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
# import pytorch_lightning as pl
# import kornia  # You can use this to get the transform and warp in this project
import math

# Don't generate pyc codes
sys.dont_write_bytecode = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def LossFn(predHomography, homography):
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

    print(predHomography[0])
    print(homography[0])
    # MSE = nn.MSELoss()
    # return MSE(homography, predHomography)
    # loss = torch.sum(torch.sub(predHomography, homography))
    # print(loss)
    # return loss
    # print(torch.linalg.norm(torch.sub(predHomography, homography), dim = 0).shape)
    # print(torch.linalg.norm(torch.sub(predHomography, homography), dim = 1).shape)
    # return torch.sum(torch.linalg.norm(homography, dim = 1))
    # print(torch.sum(torch.linalg.norm(homography, dim = 1)) / 64)
    return torch.sum(torch.linalg.norm(torch.sub(predHomography, homography), dim = 1))
    # return torch.sum(torch.linalg.norm(torch.sub(predHomography, homography) ** 2, dim = 1))


class HomographyModel(nn.Module):
    def __init__(self):
    # def __init__(self, hparams):
        super(HomographyModel, self).__init__()
        # self.hparams = hparams
        self.model = Net()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        # patchA, patchB, homography = batch
        # predHomography = self.model(patchA, patchB)
        patchs, homography = batch
        predHomography = self.model(patchs)
        loss = LossFn(predHomography, homography)
        logs = {"loss": loss}
        # return {"loss": loss, "log": logs}
        return loss

    def validation_step(self, batch):
        # patchA, patchB, homography = batch
        # predHomography = self.model(patchA, patchB)
        patchs, homography = batch
        predHomography = self.model(patchs)
        loss = LossFn(homography, predHomography)
        # img_a, patch_a, patch_b, corners, gt = batch
        # delta = self.model(patch_a, patch_b)
        # loss = LossFn(delta, img_a, patch_b, corners)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}

    def iteration_end(self, epoch, iteration, miniBatchSize, result):
        print(result)
        loss = result["val_loss"].item()
        avg_loss = loss / miniBatchSize
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
            nn.Conv2d(2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
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
            # nn.Dropout2d(0.5),

            nn.Flatten(), 
            nn.Linear(128*16*16, 1024),
            nn.ReLU(),
            # nn.Dropout(0.5),
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

    # def forward(self, xa, xb):
    def forward(self, x):
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

        # x = torch.cat((xa, xb), dim=1).to(device)

        x = x.permute(0,3,1,2).float()
        out = self.network(x)

        return out

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)