#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# Copyright (C) 2019 Hou Wei. All rights reserved.
# Project: crnn-hw
# FileName : train.py
# Author : Hou Wei
# Version : V1.0
# Date: 2020/1/13 上午10:25
# Description: crnn模型训练
# History:


import os

import numpy as np
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import CTCLoss
import torch.optim as optim
import logging
from logging import info
import time
import random

from utils import LabelConverter
import crnn as net
import params
from dataset import SyntheticChineseStringLmdbDataset

log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=log_format,
                    handlers=
                    [logging.FileHandler(
                        '../log/%s.log' % time.strftime('%Y-%m-%d')),
                        logging.StreamHandler()])

# ensure everytime the random is the same
random.seed(params.manualSeed)
np.random.seed(params.manualSeed)
torch.manual_seed(params.manualSeed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
info(device)
torch.manual_seed(1)

labelConverter = LabelConverter(params.sLabelDictPath)


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    def __init__(self, sLabelDictPath):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential()

        self.cnn.add_module("conv0", nn.Conv2d(in_channels=1,
                                               out_channels=64,
                                               kernel_size=(3, 3),
                                               padding=1))
        self.cnn.add_module("relu0", nn.ReLU(inplace=True))
        self.cnn.add_module("pooling0", nn.MaxPool2d(kernel_size=2,
                                                     stride=2))

        self.cnn.add_module("conv1", nn.Conv2d(in_channels=64,
                                               out_channels=128,
                                               kernel_size=(3, 3),
                                               padding=1))
        self.cnn.add_module("relu1", nn.ReLU(inplace=True))
        self.cnn.add_module("pooling1", nn.MaxPool2d(kernel_size=2,
                                                     stride=2))

        self.cnn.add_module("conv2", nn.Conv2d(in_channels=128,
                                               out_channels=256,
                                               kernel_size=(3, 3),
                                               padding=1))
        self.cnn.add_module("batchnorm2", nn.BatchNorm2d(num_features=256))
        self.cnn.add_module("relu2", nn.ReLU(inplace=True))

        self.cnn.add_module("conv3", nn.Conv2d(in_channels=256,
                                               out_channels=256,
                                               kernel_size=(3, 3),
                                               padding=1))
        self.cnn.add_module("relu3", nn.ReLU(inplace=True))
        self.cnn.add_module("pooling2", nn.MaxPool2d(kernel_size=(2, 2),
                                                     stride=(2, 1),
                                                     padding=(0, 1)))

        self.cnn.add_module("conv4", nn.Conv2d(in_channels=256,
                                               out_channels=512,
                                               kernel_size=(3, 3),
                                               padding=1))
        self.cnn.add_module("batchnorm4", nn.BatchNorm2d(num_features=512))
        self.cnn.add_module("relu4", nn.ReLU(inplace=True))

        self.cnn.add_module("conv5", nn.Conv2d(in_channels=512,
                                               out_channels=512,
                                               kernel_size=(3, 3),
                                               padding=1))
        self.cnn.add_module("relu5", nn.ReLU(inplace=True))
        self.cnn.add_module("pooling3", nn.MaxPool2d(kernel_size=(2, 2),
                                                     stride=(2, 1),
                                                     padding=(0, 1)))

        self.cnn.add_module("conv6", nn.Conv2d(in_channels=512,
                                               out_channels=512,
                                               kernel_size=(2, 2),
                                               padding=0))
        self.cnn.add_module("batchnorm6", nn.BatchNorm2d(num_features=512))
        self.cnn.add_module("relu6", nn.ReLU(inplace=True))

        sAllLabel = "".join(open(sLabelDictPath).read().splitlines())
        iLabelCount = len(sAllLabel) + 1

        self.rnn = nn.Sequential()
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 256, 256),
            BidirectionalLSTM(256, 256, iLabelCount))

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()

        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c] 71, 1, 512

        output = self.rnn(conv)
        output = F.log_softmax(output, dim=2)

        return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(net, criterion, optimizer, iterData):
    for parameter in net.parameters():
        parameter.requires_grad = True

    net.apply(weights_init)
    net.train()

    dcData = iterData.next()

    ptImg = dcData["image"]
    ptLabel = dcData["labelNum"]
    npAddDimImg = np.reshape(
        ptImg, (ptImg.shape[0], 1, ptImg.shape[1], ptImg.shape[2]))
    # info("npAddDimImg",npAddDimImg.shape)

    npAddDimImg = npAddDimImg.float()
    npAddDimImg = npAddDimImg.cuda(device)
    preds = net.forward(npAddDimImg)
    # info("preds.shape", preds.shape)

    # @TODO:处理变长文字
    # info("preds.shape", preds.shape)
    # info("ptLabel", ptLabel)
    # info("ptLabel.shape", ptLabel[0].shape)

    if (ptLabel.shape[0] < params.iBatchSize):
        input_lengths = torch.tensor([preds.size(0)] * ptLabel.shape[0])
        label_lengths = torch.tensor([0] * ptLabel.shape[0])
    else:
        input_lengths = torch.tensor([preds.size(0)] * params.iBatchSize)
        label_lengths = torch.tensor([0] * params.iBatchSize)
    # info("input_lengths", input_lengths)
    # info("label_lengths", label_lengths)

    for i in range(ptLabel.shape[0]):
        label_lengths[i] = ptLabel[i].shape[0]

    ptLabel = ptLabel.cuda(device)

    # print(preds.shape)
    # print(ptLabel)
    # print(input_lengths)
    # print(label_lengths)
    cost = criterion(preds, ptLabel, input_lengths, label_lengths)
    cost.backward()
    optimizer.step()
    # info(cost)
    return cost


def validate(net, criterion, testLoader):
    for p in net.parameters():
        p.requires_grad = False
    net.eval()
    iterData = iter(testLoader)
    iDataLen = len(iterData)

    fCostSum = 0

    for i in range(iDataLen):
        dcData = iterData.next()
        ptImg = dcData["image"]
        ptLabel = dcData["labelNum"]
        lChnLabel = dcData["labelChn"]
        ptLabel = ptLabel.cuda()

        # info("npImg.shape",ptImg.shape)
        # info(lChnLabel)
        npAddDimImg = np.reshape(
            ptImg, (ptImg.shape[0], 1, ptImg.shape[1], ptImg.shape[2]))
        # info("npAddDimImg",npAddDimImg.shape)
        npAddDimImg = npAddDimImg.float()
        npAddDimImg = npAddDimImg.cuda(device)
        preds = net.forward(npAddDimImg)

        if (ptLabel.shape[0] < params.iBatchSize):
            input_lengths = torch.tensor([preds.size(0)] * ptLabel.shape[0])
            label_lengths = torch.tensor([0] * ptLabel.shape[0])
        else:
            input_lengths = torch.tensor([preds.size(0)] * params.iBatchSize)
            label_lengths = torch.tensor([0] * params.iBatchSize)

        cost = criterion(preds, ptLabel, input_lengths, label_lengths)
        fCostSum += cost

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).cpu().numpy()
    # print(preds)

    for iIndex in range(10 if (10 < preds.shape[0]) else preds.shape[0]):
        ptPreds = preds[iIndex].copy()
        sPred = labelConverter.num2char(preds[iIndex])
        sStrictPred = labelConverter.num2strict_char(ptPreds)
        info("%-20s => %-20s, ground_truth: %20s" %
             (sPred, sStrictPred, lChnLabel[iIndex]))

    return fCostSum / iDataLen


def main():
    sSavePath = "../model/crnn-hw.pth"
    # datasetTrain = SyntheticChineseStringDataset(params.sTrainData,
    #                                              params.sImgDir)
    datasetTrain = SyntheticChineseStringLmdbDataset("../data/train_lmdb",
                                                     params.sLabelDictPath)
    trainLoader = DataLoader(dataset=datasetTrain,
                             batch_size=params.iBatchSize,
                             shuffle=False,
                             num_workers=4)

    datasetTest = SyntheticChineseStringLmdbDataset("../data/test_lmdb_mini",
                                                    params.sLabelDictPath)
    testLoader = DataLoader(dataset=datasetTest,
                            batch_size=params.iBatchSize,
                            shuffle=True,
                            num_workers=4)

    crnn = CRNN(params.sLabelDictPath)
    # crnn = net.CRNN(params.iImgH, 1, 5995, 256)

    info(crnn)

    # npData = datasetTrain[0]["image"]
    # info(npData.shape)

    criterion = CTCLoss()
    criterion = criterion.cuda()
    # optimizer = optim.Adam(crnn.parameters(), lr=params.lr,
    #                        betas=(params.beta1, 0.999))
    optimizer = optim.RMSprop(crnn.parameters(), lr=params.lr)

    # train
    info("Start training...")
    crnn.to(device)
    for iEpoch in range(params.iNEpoch):
        iterTrainData = iter(trainLoader)
        i = 0
        while (i < len(trainLoader)):
            cost = train(crnn, criterion, optimizer, iterTrainData)
            i += 1

            if (i % params.iDisplayInterval == 0):
                info("[Epoch %d/%d], [Iter %d/%d], Train Loss:%f" %
                     (iEpoch, params.iNEpoch, i, len(trainLoader), cost))

            if (i % params.iValInterval == 0):
                info("Start Val")
                info("[Epoch %d/%d], [Iter %d/%d], Test Loss:%f\n" %
                     (iEpoch, params.iNEpoch, i, len(trainLoader),
                      validate(crnn, criterion, testLoader)))

            if (i % params.iSaveInterval == 0):
                torch.save(crnn.state_dict(),
                           '{0}/netCRNN_{1}_{2}.pth'.format(params.sSaveDir,
                                                            iEpoch, i))


if __name__ == "__main__":
    main()
