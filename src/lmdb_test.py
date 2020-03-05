#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# Copyright (C) 2019 Hou Wei. All rights reserved.
# Project: crnn-hw
# FileName : train.py
# Author : Hou Wei
# Version : V1.0
# Date: 2020/1/13 上午10:25
# Description: 测试lmdb
# History:

import lmdb
import numpy as np

#
# env = lmdb.open("../data/train", map_size=1099511627776)
#
# txn = env.begin(write=True)
#
#
# txn.put(key = '1'.encode(), value = 'aaa'.encode())
# txn.put(key = '2'.encode(), value = 'bbb'.encode())
# txn.put(key = '3'.encode(), value = 'ccc'.encode())
#
# txn.commit()
#
# txn = env.begin()
#
# print(txn.get(str(2).encode()))
#
# env.close()

x = np.array([1, 2, 3])
print(x.dtype)
