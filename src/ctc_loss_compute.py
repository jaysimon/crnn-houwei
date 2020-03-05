#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# Copyright (C) 2019 Hou Wei. All rights reserved.
# Project: 07-crnn-hw
# FileName : ctc_loss_compute.py
# Author : Hou Wei
# Version : V1.0
# Date: 2020/2/25 下午3:17
# Description: ctc_loss_compute
# History:

import numpy as np

import torch
from torch.nn import CTCLoss

input_lengths = torch.tensor([71] * 1)
label_lengths = torch.tensor([10] * 1)
# preds = torch.tensor(np.zeros((71, 1, 5990)))
preds = np.load("../data/preds.npy")
preds = torch.tensor(preds)

ptLabel = torch.tensor(
    np.array([263, 82, 29, 56, 35, 435, 890, 293, 126, 129]).reshape((1,10)))

criterion = CTCLoss(zero_infinity=True)

print("preds", preds.shape)
print("ptLabel", ptLabel.shape)
print("input_lengths", input_lengths.shape)
print(input_lengths)
print("label_lengths", label_lengths.shape)
print(label_lengths)

cost = criterion(preds, ptLabel, input_lengths, label_lengths)
print("cost", cost)

_ ,preds = preds.max(2)
print(preds)
