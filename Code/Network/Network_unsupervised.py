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

def SolveDLT(cornersA, cornersB):
    batchSize = cornersA.size()[0]
    b = torch.zeros(batchSize, 8)
    A = torch.zeros(batchSize, 8, 8)
    h8 = torch.ones(batchSize, 9)

    # fill b and A
    for i in range(batchSize):
        for j in range(4):
            fir = 2 * j
            sec = 2 * j + 1

            xa = cornersA[i][j][0]
            ya = cornersA[i][j][1]
            xb = cornersB[i][j][0]
            yb = cornersB[i][j][1]

            b[i][fir] = -yb
            b[i][sec] = xb

            A[i][fir][3:6] = torch.tensor([-xa, -ya, -1])
            A[i][fir][6:8] = torch.tensor([yb * xa, yb * ya])
            A[i][sec][0:3] = torch.tensor([xa, ya, 1])
            A[i][sec][6:8] = torch.tensor([-xb * xa, -xb * ya])

        Ainv = torch.inverse(A[i])
        h8[i, 0:8] = torch.matmul(Ainv, b[i])

    return h8.view(-1, 3, 3)

def LossFn(predH4Pts, corners, imgA, patchB, iter, h4pt, homo):
# def LossFn(predH4Pts, corners, patchA, patchB, iter, h4pt, homo):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################
    print(predH4Pts[0])
    print(h4pt[0])
    batchSize = predH4Pts.size()[0]

    cornersB = corners + predH4Pts.view(-1, 4, 2)
    H = SolveDLT(corners, cornersB).to(device)
    Hinv = torch.inverse(H)


    patchAwarpCrop = torch.zeros(batchSize, 1, 128, 128).to(device)
    for i in range(batchSize):
        x = int(corners[i][0][0].item())
        y = int(corners[i][0][1].item())
        img = torch.tensor(imgA[i]).float().to(device).unsqueeze(0).unsqueeze(0)

        patchAwarp = kornia.geometry.transform.warp_perspective(img, Hinv[i].unsqueeze(0), (img.size()[-2], img.size()[-1]))
        patchAwarpCrop[i, 0, :, :] = patchAwarp[0, 0, y:y+128, x:x+128]

    # img_both = cv2.hconcat([copy.copy(patchAwarpCrop[0][0]).cpu().detach().numpy(), copy.copy(patchB[0][0]).cpu().detach().numpy()])
    # cv2.imwrite("../Data/patchAwarpB{}.jpg".format(iter), img_both)

    patchB = torch.unsqueeze(patchB, dim = 1)

    loss = nn.L1Loss()
    return loss(patchAwarpCrop, patchB)


class HomographyModel(nn.Module):
    def __init__(self):
        super(HomographyModel, self).__init__()
        self.model = Net()

    def forward(self, xa, xb):
        return self.model(xa, xb)

    def training_step(self, batch, iter):
        patchA, patchB, h4pt, corners, img, homography = batch
        predHomography = self.model(patchA, patchB)
        # patchs, homography = batch
        # predHomography = self.model(patchs)
        loss = LossFn(predHomography, corners, img, patchB, iter, h4pt, homography)
        # loss = LossFn(predHomography, corners, patchA, patchB, iter, h4pt, homography)
        logs = {"loss": loss}
        # return {"loss": loss, "log": logs}
        return loss

    def validation_step(self, batch, iter):
        patchA, patchB, h4pt, corners, img, homography = batch
        predHomography = self.model(patchA, patchB)
        # patchs, homography = batch
        # predHomography = self.model(patchs)
        loss = LossFn(predHomography, corners, img, patchB, iter, h4pt, homography)
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
        # x = torch.cat((xa, xb), dim=1)
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