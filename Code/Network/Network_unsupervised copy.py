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
import kornia  # You can use this to get the transform and warp in this project
import math
import cv2
import copy

# Don't generate pyc codes
sys.dont_write_bytecode = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def SolveDLT(cornersA, predH4Pts):
    cornersB = cornersA + predH4Pts

    batchSize = cornersA.size()[0]
    b = torch.zeros(batchSize, 8)
    A = torch.zeros(batchSize, 8, 8)
    h8 = torch.zeros(batchSize, 8)

    # fill b and A
    for i in range(batchSize):
        for j in range(4):
            fir = 2 * j
            sec = 2 * j + 1
            b[i][fir] = -cornersB[i][sec]
            b[i][sec] = cornersB[i][fir]

            A[i][fir][3:6] = torch.tensor([-cornersA[i][fir], -cornersA[i][sec], -1])
            A[i][fir][6:8] = torch.tensor([cornersB[i][sec] * cornersA[i][fir],
                            cornersB[i][sec] * cornersA[i][sec]])
            A[i][sec][0:3] = torch.tensor([cornersA[i][fir], cornersA[i][sec], 1])
            A[i][sec][6:8] = torch.tensor([-cornersB[i][fir] * cornersA[i][fir],
                            -cornersB[i][fir] * cornersA[i][sec]])

        Ainv = torch.inverse(A[i])
        h8[i] = torch.matmul(Ainv, b[i])

    return h8

def LossFn(predH4Pts, corners, imgA, patchB, iter):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################

    cornersB = corners + predH4Pts.reshape(predH4Pts.size()[0], 4, 2)
    # print(corners.size())
    # print(cornersB.size())
    corners = corners - corners[:, 0].view(-1, 1, 2)

    # corners = torch.unsqueeze(corners, dim = 1)
    # cornersB = torch.unsqueeze(cornersB, dim = 1)
    # print(corners)
    # print(cornersB)

    H = kornia.geometry.transform.get_perspective_transform(corners, cornersB)

    Hinv = torch.inverse(H)
    # print(Hinv.size())
    # imgA = torch.unsqueeze(imgA, dim = 1)
    for i in range(predH4Pts.size()[0]):
        img = torch.tensor(imgA[i]).unsqueeze(0).unsqueeze(0).to(device)
        # print(img.size())

        patchAwarp = kornia.geometry.transform.warp_perspective(img, Hinv[i].unsqueeze(0), (128, 128))

        # if i == 
        cv2.imwrite("../Data/patchAwarp{}.jpg".format(i), copy.copy(patchAwarp).cpu().detach().numpy())
        # cv2.imwrite("../Data/patchA{}.jpg".format(iter), copy.copy(patchA[0][0]).cpu().detach().numpy())
        cv2.imwrite("../Data/patchB{}.jpg".format(i), copy.copy(patchB[i]).cpu().detach().numpy())
    # print(patchAwarp.size())
    # print(patchA.size())
    
    return torch.nn.functional.l1_loss(patchAwarp, patchB)


class HomographyModel(nn.Module):
    def __init__(self):
    # def __init__(self, hparams):
        super(HomographyModel, self).__init__()
        # self.hparams = hparams
        self.model = Net()

    # def forward(self, x):
    #     return self.model(x)

    def training_step(self, batch, iter):
        patchA, patchB, homography, corners, img = batch
        predHomography = self.model(patchA, patchB)
        # patchs, homography = batch
        # predHomography = self.model(patchs)
        loss = LossFn(predHomography, corners, img, patchB, iter)
        logs = {"loss": loss}
        # return {"loss": loss, "log": logs}
        return loss

    def validation_step(self, batch, iter):
        patchA, patchB, homography, corners, img = batch
        predHomography = self.model(patchA, patchB)
        # patchs, homography = batch
        # predHomography = self.model(patchs)
        loss = LossFn(predHomography, corners, img, patchB, iter)
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

    def forward(self, xa, xb):
    # def forward(self, x):
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

        x = torch.stack((xa, xb), dim=1)
        # print(xa.size(), xb.size(), x.size())
        # x = x.permute(0,3,1,2).float()
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