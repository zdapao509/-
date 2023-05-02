"""Normalize features"""
'''
标准差标准化 StandardScaler

使用均值与方差，对服从正态分布的数据处理，得到符合标准正态分布的数据

处理方法：标准化数据减去均值，然后除以标准差，经过处理后数据符合标准正态分布，即均值为0，标准差为1；
转化函数：x = (x-mean) / std；
适用性：适用于本身服从正态分布的数据；
'''
import numpy as np


def normalize(features):

    features_normalized = np.copy(features).astype(float)

    # 计算均值
    features_mean = np.mean(features, 0)# 压缩行，对各列求均值

    # 计算标准差
    features_deviation = np.std(features, 0)# 压缩行，对各列标准差

    # 标准化操作
    if features.shape[0] > 1:#矩阵行数：shape[0]；行数只有一行的话就保留原数据，不然减掉均值的话就会全为0
        features_normalized -= features_mean
        # print(features_normalized)

    # 防止除以0
    features_deviation[features_deviation == 0] = 1
    # print(features_deviation)

    # features_deviation[True] = 1        #[[1. 1. 1.]]
    # print(features_deviation)

    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation

# '验证：'
# num1 = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,79]])
# #创建ndarray数组; np.array产生numpy.ndarray类型的数据，不能进行矩阵意义上的乘法。np.matrix，np.mat产生numpy.matrix类型数据，可以进行矩阵相乘
# print(type(num1))
# num2 = np.mat(num1)
# [a,b,c]=normalize(num2)
# print(a)
# print(b)
# print(c)