#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# Copyright (C) 2019 Hou Wei. All rights reserved.
# Project: 07-crnn-hw
# FileName : dataset.py
# Author : Hou Wei
# Version : V1.0
# Date: 2020/3/4 上午10:56
# Description: 读取lmdb数据集,以及相关处理
# History:

import lmdb
import sys
import logging
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

import params
from utils import LabelConverter


class SyntheticChineseStringLmdbDataset(Dataset):
    """
    处理中文合成数据集的lmdb数据类，为训练做准备
    """

    def __init__(self, sLmdbPath, sLabelDictPath):
        self.env = lmdb.open(
            sLmdbPath,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        if (not self.env):
            logging.info("cannot create lmdb from %s" % (sLmdbPath))
            sys.exit(0)
        with self.env.begin(write=False) as txn:
            self.iSampleCount = int(txn.get("iSampleCount".encode("utf-8")))
        self.lLabelDict = open(sLabelDictPath).read().splitlines()
        self.labelConverter = LabelConverter(params.sLabelDictPath)

    def __len__(self):
        return self.iSampleCount

    def __getitem__(self, iIndex):
        assert iIndex <= len(self) and iIndex >= 0, "index out of range"
        with self.env.begin(write=False) as txn:
            sImageKey = "image-%09d" % iIndex
            sLabelChnKey = "labelchn-%09d" % iIndex
            sLabelNumKey = "labelnum-%09d" % iIndex
            sNameKey = "name-%09d" % iIndex
            # print(txn.get(sNameKey.encode()))
            binImage = txn.get(sImageKey.encode())
            bufImage = np.frombuffer(binImage, dtype=np.uint8)
            matImg = cv2.imdecode(bufImage, cv2.IMREAD_GRAYSCALE)
            if (params.bNormalize):
                matImg = self.__reshape_normalize__(matImg)
            sLabelNum = txn.get(sLabelNumKey.encode()).decode()
            sLabelChn = txn.get(sLabelChnKey.encode()).decode()
            # npLabel = self.__encode__(sLabelNum)
            npLabel = self.labelConverter.char2num(sLabelChn)
            # print(sLabelNum)
            sName = txn.get(sNameKey.encode()).decode()

            sample = {'image': matImg, 'labelNum': npLabel,
                      'labelChn': sLabelChn, "name": sName}
            # print("sample:", matImg.shape, "label", npLabel.shape, npLabel.dtype)
        return sample

    def __reshape_normalize__(self, matImg):
        """
        对图像进行归一化
        :param matImg:
        :return:
        """
        matImg = cv2.resize(matImg, (params.iImgW, params.iImgH))
        matImg = ((matImg / 255.0) - 0.5) * 2
        return matImg

    def __encode__(self, sLabel):
        """
        转换string 数字标签为数字list
        :param sLabel:[123 324 423]
        :return: npLabel:[123, 324, 423]
        """
        lWords = sLabel.split(",")
        lCharacter = []
        for sWord in lWords:
            lCharacter.append(int(sWord))
        npLabel = np.array(lCharacter)
        return npLabel


def main():
    labelConverter = LabelConverter(params.sLabelDictPath)
    dataset = SyntheticChineseStringLmdbDataset("../data/train_lmdb",
                                                params.sLabelDictPath)
    for i in range(19):
        print(labelConverter.num2char(dataset[i]["labelNum"]))
        print(labelConverter.char2num(dataset[i]["labelChn"]))
        print(labelConverter.num2char(
            labelConverter.char2num(dataset[i]["labelChn"])))
        print(dataset[i]["labelChn"])
        print("")


if __name__ == "__main__":
    main()
