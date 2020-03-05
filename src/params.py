#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# Copyright (C) 2019 Hou Wei. All rights reserved.
# Project: 07-crnn-hw
# FileName : params.py
# Author : Hou Wei
# Version : V1.0
# Date: 2020/2/14 下午3:11
# Description: params
# History:

######## 数据参数 ########
# 训练数据列表路径
sTrainData = "/home/houwei/02-data/SyntheticChineseStringDataset/train.txt"
#测试数据列表路径
sTestData = "/home/houwei/02-data/SyntheticChineseStringDataset/test_mini.txt"
# 汉字数字对应表路径
sLabelDictPath = "/home/houwei/02-data/SyntheticChineseStringDataset/label.txt"
# 图片数据路径
sImgDir = "/home/houwei/02-data/SyntheticChineseStringDataset/images"
# 对图像进行归一化
bNormalize = True
# 图像高度
iImgH = 32
# 图像宽度
iImgW = 100
# 随机种子
manualSeed = 1234
######## 训练参数 ########
# batchsize
iBatchSize = 64
# 每多少次迭代显示依次结果
iDisplayInterval = 100
# 每多少次迭代进行测试
iValInterval = 1000
# 每多少次存储模型
iSaveInterval = 5000
# 运行epoch个数
iNEpoch = 100
# 学习率
lr = 0.00001
# beta1
beta1 = 0.5 # beta1 for adam. default=0.5
# 存储路径
sSaveDir="../model"

