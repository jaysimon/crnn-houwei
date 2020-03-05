#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# Copyright (C) 2019 Hou Wei. All rights reserved.
# Project: 07-crnn-hw
# FileName : test.py
# Author : Hou Wei
# Version : V1.0
# Date: 2020/2/14 下午3:54
# Description: test
# History:

import torch

T = 50      # Input sequence length
C = 20      # Number of classes (including blank)
N = 16      # Batch size
S = 30      # Target sequence length of longest target in batch
S_min = 10  # Minimum target length, for demonstration purposes

# Initialize random batch of input vectors, for *size = (T,N,C)
input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
print("input", input.shape)
# Initialize random batch of targets (0 = blank, 1:C = classes)
target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)
print("target:", target.shape)
input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)

print("input_lengths",input_lengths)

print("target_lengths",target_lengths)

ctc_loss = torch.nn.CTCLoss()
loss = ctc_loss(input, target, input_lengths, target_lengths)
loss.backward()
print(loss)