# Created by [Yuexiong Ding] at 2018/3/14
# 数据清理
#

import numpy as np
import pandas as pd

# 原始训练数据路径
Raw_TrainData_Path = r'../DataSet/TrainData/raw_train_20180301.txt'
# 原始测试数据路径
Raw_TestData_Path = r'../DataSet/TestData/raw_test_a_20180301.txt'

# 读取数据
# RawData = np.loadtxt(Raw_TrainData_Path, dtype=str)
RawData = pd.read_csv(Raw_TrainData_Path, sep=" ")

print(RawData)

