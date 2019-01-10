# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import math
from ml.linerRegession import standRegession


def lwlr(testPoint, dataMat, labelMat, k):
    """
     Description：
            局部加权线性回归，在待预测点附近的每个点赋予一定的权重，在子集上基于最小均方差来进行普通的回归。
        Args：
            testPoint：样本点
            xArr：样本的特征数据，即 feature
            yArr：每个样本对应的类别标签，即目标变量
            k:关于赋予权重矩阵的核的一个参数，与权重的衰减速率有关
        Returns:
            testPoint * ws：数据点与具有权重的系数相乘得到的预测点
        Notes:
            这其中会用到计算权重的公式，w = e^((x^((i))-x) / -2k^2)
            理解：x为某个预测点，x^((i))为样本点，样本点距离预测点越近，贡献的误差越大（权值越大），越远则贡献的误差越小（权值越小）。
            关于预测点的选取，在我的代码中取的是样本点。其中k是带宽参数，控制w（钟形函数）的宽窄程度，类似于高斯函数的标准差。
            算法思路：假设预测点取样本点中的第i个样本点（共m个样本点），遍历1到m个样本点（含第i个），算出每一个样本点与预测点的距离，
            也就可以计算出每个样本贡献误差的权值，可以看出w是一个有m个元素的向量（写成对角阵形式）。
    :param dataMat: 特征数据
    :param labelMat:  目标变量
    :param k: 带宽参数
    :return:
    """
    xMat = np.mat(dataMat)
    yMat = np.mat(labelMat).T
    length = xMat.shape[0]
    weights = np.eye(length)
    for j in range(length):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = math.exp(diffMat * diffMat.T / (-2 * (k ** 2)))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print('行列式为0，不可求逆')
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(xArr, xMat, yMat, k=1.0):
    '''
        Description：
            测试局部加权线性回归，对数据集中每个点调用 lwlr() 函数
        Args：
            testArr：测试所用的所有样本点
            xArr：样本的特征数据，即 feature
            yArr：每个样本对应的类别标签，即目标变量
            k：控制核函数的衰减速率
        Returns：
            yHat：预测点的估计值
    '''
    xMat = np.mat(xMat)
    length = xMat.shape[0]
    yHat = np.zeros(length)
    for i in range(length):
        yHat[i] = lwlr(xArr[i], xMat, yMat, k)
    return yHat


def lwlr_plot(xMat, yMat, yHat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter([xMat[:, 1].flatten().A[0]], [yMat.flatten().A[0]], s=2, c='red')
    ax.scatter([xMat[:, 1].flatten().A[0]], yHat, s=2, c='black')
    plt.show()


if __name__ == '__main__':
    dm, lm = standRegession.load_data_file('data.txt')
    yhat = lwlrTest(dm, dm, lm, 1.0)
    print(yhat)
    lwlr_plot(np.mat(dm), np.mat(lm).T, yhat)
