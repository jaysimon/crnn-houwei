#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# Copyright (C) 2019 Hou Wei. All rights reserved.
# Project: 07-crnn-hw
# FileName : utils.py
# Author : Hou Wei
# Version : V1.0
# Date: 2020/3/4 上午9:36
# Description: 本项目中的常用函数
# History:

import numpy as np
import params


class LabelConverter():
    def __init__(self, sLabelDictPath):
        """
        利用汉字列表进行初始化
        :param sLabelDictPath: "/SyntheticChineseStringDataset/label.txt"
        """
        self.lNum2Char = open(sLabelDictPath).read().splitlines()
        self.dcChar2Num = {}
        for i in range(len(self.lNum2Char)):
            self.dcChar2Num[self.lNum2Char[i]] = i

    def char2num(self, sChn):
        """
        转换汉字为其对应的序号
        :param sChn: "你好啊!"
        :return: npLabel: [ 145  128 1343  293]
        """
        lNum = []
        for sChar in sChn:
            lNum.append(int(self.dcChar2Num[sChar]))
        npLabel = np.array(lNum)
        npLabel += 1
        return npLabel

    def num2char(self, npLabel):
        """
        将numpy数字转换为对应的汉字
        :param npLabel: [ 145  128 1343  293]
        :return: "你好啊!"
        """
        npLabel -= 1
        sChn = ""
        for i in range(npLabel.shape[0]):
            iIndex = npLabel[i]
            if (iIndex == -1):
                sChn += "-"
            else:
                sChn += self.lNum2Char[iIndex]
        return sChn

    def num2strict_char(self, npLabel):
        """
        转换稀疏汉字序列为紧密汉字
        :param npLabel: [ 145  0 128 0 1343 0 0  293]
        :return: "你好啊!"
        """
        npLabel -= 1
        sChn = ""
        for i in range(npLabel.shape[0]):
            iIndex = npLabel[i]
            if (iIndex != -1):
                sChn += self.lNum2Char[iIndex]
        return sChn


def main():
    labelConverter = LabelConverter(params.sAllLabelPath)

    npChn = labelConverter.char2num("你好啊!")
    print(npChn)
    sChn = labelConverter.num2char(npChn)
    print(sChn)
    npChn = np.array([145, 128, 1343, 0, 0, 293])
    print(labelConverter.num2char(npChn))
    print(labelConverter.num2strict_char(npChn))


if __name__ == "__main__":
    main()
