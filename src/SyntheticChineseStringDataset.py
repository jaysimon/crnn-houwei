#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# Copyright (C) 2019 Hou Wei. All rights reserved.
# Project: crnn-hw
# FileName : SyntheticChineseStringDataset.py
# Author : Hou Wei
# Version : V1.0
# Date: 2020/1/13 上午10:25
# Description: 转换创建中文lmdb数据集
# History:

import os
import cv2
import sys
import pandas as pd
import time
import logging
import shutil
import lmdb
import numpy as np
from torch.utils.data import Dataset, DataLoader

import params

log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=log_format,
                    handlers=
                    [logging.FileHandler(
                        '../log/%s.log' % time.strftime('%Y-%m-%d')),
                        logging.StreamHandler()])


class SyntheticChineseStringDataset(Dataset):
    """
    处理中文合成数据集类，为训练做准备,用于创建lmdb数据，也可以作为torch的数据输入
    """

    def __init__(self, sCsvPath, sImgPath, sLabelDictPath):
        self.lData = open(sCsvPath).read().splitlines()
        self.sImgPath = sImgPath
        self.lAllLabel = open(sLabelDictPath).read().splitlines()
        # self.show_character(self.extract_data(self.lTrain[0])["Label"])
        # print(self.__getitem__(0))

    def __len__(self):
        return len(self.lData)

    def __getitem__(self, iIndex):
        dcData = self.extract_data(self.lData[iIndex])

        matImg = cv2.imread(os.path.join(self.sImgPath, dcData["name"]),
                            cv2.IMREAD_GRAYSCALE)
        if (params.bNormalize):
            matImg = self.__normalize__(matImg)

        npLabel = np.array(dcData["label"])
        sample = {'image': matImg, 'label': npLabel}
        return sample

    def __normalize__(self, matImg):
        """
        对图像进行归一化
        :param matImg:
        :return:
        """
        matImg = matImg / 255.0
        return matImg

    def extract_data(self, sLine):
        """
        解析路径和标记数据
        :param sLine:"20455828_2605100732.jpg 263 82 29 56 35 435 890 293 126 129"
        :return:{'name': '20455828_2605100732.jpg', 'label': [82, 29, 56, 35, 435, 890, 293, 126, 129]}
        """
        lWords = sLine.split()
        dcData = {}
        dcData["name"] = lWords[0]
        lCharacter = []
        for sWord in lWords[1:]:
            lCharacter.append(sWord)
        dcData["label"] = lCharacter
        return dcData

    def get_character(self, lLabels):
        lText = []
        for iLabel in lLabels:
            lText.append(self.lAllLabel[int(iLabel)])
        return "".join(lText)

    def writeCache(self, env, cache):
        """
        写cache到lmdb数据中，同时转换格式
        :param cache:
        :return:
        """
        with env.begin(write=True) as txn:
            for k, v in cache.items():
                if (type(k) == str):
                    k = k.encode()
                if (type(v) == str):
                    v = v.encode()
                txn.put(k, v)

    def create_lmdb_data(self, sSavePath):
        """
        转换数据为lmdb格式
        :param sSavePath: lmdb存储路径
        :return:
        """
        if (os.path.exists(sSavePath)):
            # shutil.rmtree(sSavePath)
            # os.mkdir(sSavePath)

            # 防止数据覆盖,请手动删除数据
            logging.info("%s already exist!" % sSavePath)
            return 0
        else:
            os.mkdir(sSavePath)

        iDataLen = len(self.lData)
        env = lmdb.open(sSavePath, map_size=1099511627776)
        dcCache = {}
        iIndex = 0
        for sLine in self.lData:
            dcData = self.extract_data(sLine)
            sImgPath = os.path.join(self.sImgPath, dcData["name"])
            if (not os.path.exists(sImgPath)):
                logging.info("%s does not exist!" % sImgPath)
            with open(sImgPath, "rb") as f:
                byteImg = f.read()
            imageKey = "image-%09d" % iIndex
            labelChnKey = "labelchn-%09d" % iIndex
            labelNumKey = "labelnum-%09d" % iIndex
            nameKey = "name-%09d" % iIndex

            dcCache[imageKey] = byteImg
            dcCache[nameKey] = dcData["name"]

            sCharLabel = self.get_character(dcData["label"])
            dcCache[labelChnKey] = sCharLabel
            dcCache[labelNumKey] = ",".join(dcData["label"])

            if (iIndex % 1000 == 0):
                self.writeCache(env, dcCache)
                dcCache = {}
                logging.info("Written %d in %d, finished: %.2f %%" %
                             (iIndex, iDataLen, 100.0 * iIndex / iDataLen))
            iIndex += 1
        iDataLen = iIndex
        dcCache["iSampleCount"] = str(iDataLen)
        self.writeCache(env, dcCache)
        env.close()
        logging.info("Create %s dataset with %d Sample" % (sSavePath, iDataLen))




def main():
    dataset = SyntheticChineseStringDataset(params.sTrainData,
                                            params.sImgDir,
                                            params.sLabelDictPath)
    dataset.create_lmdb_data("../data/train_lmdb")

    dataset = SyntheticChineseStringDataset(params.sTestData,
                                            params.sImgDir,
                                            params.sLabelDictPath)
    dataset.create_lmdb_data("../data/test_lmdb_mini")

    logging.info("Init dataset success!")

if __name__ == "__main__":
    main()
