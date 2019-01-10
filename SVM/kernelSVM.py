#!/usr/bin/python
# -*- coding: UTF-8 -*-

#   data-analysis
#   kernelSVM.py

"""
    Description: 核函数+软正则化
    author : zhangjingbo 
    Date:    2018/12/24
"""
import numpy as np
import random


class KernelNotSetException(Exception):
    """
    自定义核函数未设置异常
    """

    def __init__(self, code=100, message='核函数未正确配置'):
        self.code = code
        self.message = message


class KernelSVM:
    """
    核函数+软正则实现支持向量：
        初始化时类仅需定义数据集，惩罚因子C，松弛变量zeta
    """

    def __init__(self, C, zeta, data_matrix, label_matrix):
        self.C = C
        self.zeta = zeta
        self.data_matrix = data_matrix
        self.label_matrix = label_matrix

    def kernel_fun(self, kernel_fun):
        """
        设置核函数
        :param kernel_fun:
        :param params:
        """
        self.kernel_fun = kernel_fun

    def calc_Ek(self, alphas, labelMat, k, b):
        """
        计算预测值
        """
        fXk = float(np.multiply(alphas, labelMat).T * self.cacl_kernel(self.data_matrix, self.data_matrix[k, :]) + b)
        Ek = fXk - float(labelMat[k])
        return Ek

    def cacl_kernel(self, xi, xj):
        """
        计算核函数的值
        :param xi:
        :param xj:
        :return:
        """
        if self.kernel_fun is None:
            raise KernelNotSetException()
        k = self.kernel_fun(xi, xj)
        return k

    def selectJ(self, i, Ei):  # this is the second choice -heurstic, and calcs Ej
        pass

    def smo_optimization(self, maxIter):
        """
        Args:
            maxIter 退出前最大的循环次数
        Returns:
            b       模型的常量值
            alphas  拉格朗日乘子
        """
        m, n = self.data_matrix.shape
        featureData = self.data_matrix
        labelData = self.label_matrix
        b = 0
        iter = 0
        alphas = np.mat(np.zeros([m, 1]))
        while iter < maxIter:
            alphaPairsChanged = 0
            for i in range(m):
                fxi = np.multiply(alphas, labelData).T * featureData * featureData[i, :].T + b
                Ei = fxi - float(labelData[i])


class KernelFunction:
    """
    定义常用核函数
    """

    def __init__(self, **params):
        """
        使用rbf核、拉普拉斯核、多项核时传入参数
        :param params:
        """
        self.params = params

    def lr_kernel(self, xi, xj):
        """
        线性svm核函数：k(xi,xj)= xi*xj.T
        :param xi:
        :param xj:
        :return:
        """
        lrk = xi * xj.T
        return lrk

    def rbf_kernel(self, xi, xj):
        """
        定义高斯核的计算公式:k(xi,xj)=exp(-(xi-xj)^2/2theta^2)
        :param theta:
        :param xi:
        :param xj:
        :return:
        """
        theta = self.params['theta']
        rbfk = np.exp(-(xi - xj) * (xi - xj).T / (2 * theta ** 2))
        return rbfk

    def poly_kernel(self, xi, xj):
        """
        定义多项式核计算公式: k(xi,xj) = (xi^T*xj)^d
        :param d:
        :param xi:
        :param xj:
        :return:
        """
        d = self.params['d']
        polyk = (xi * xj.T) ** d
        return polyk

    def laplace_kernel(self, xi, xj):
        """
        拉普拉斯核：k(xi,xj)=exp(-(xi-xj)/theta)
        :param theta:
        :param xi:
        :param xj:
        :return:
        """
        theta = self.params['theta']
        xminus = xi - xj
        l1 = np.sum(np.abs(xminus), axis=1)
        laplacek = np.exp(-l1 / theta)
        return laplacek


if __name__ == '__main__':
    xi = np.mat([1, 2, 3])
    xj = np.mat([0, 1, 2])
