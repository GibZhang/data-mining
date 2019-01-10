#!/usr/bin/python
# -*- coding: UTF-8 -*-

#   data-analysis
#   basicSVM.py

"""
    Description: 基础版SVM
    author : zhangjingbo 
    Date:    2018/12/20
"""
import os
import numpy as np
import pickle
import matplotlib.pylab as plt
import random


class basicSVM:
    def __init__(self, testSet='testSet'):
        self.testSet = testSet
        if not os.path.isfile(testSet):
            self.file2vect()
            self.save_data()
        self.featureMat = self.load_data()

    def file2vect(self):
        featureMat = []
        try:
            with open(self.testSet + '.txt') as f:
                for s in f:
                    tmp = s.rstrip('\n').split('\t')
                    featureMat.append(tmp)
        except FileNotFoundError as e:
            print(e.filename)
        self.featureMat = np.mat(featureMat).astype(float)

    def load_data(self):
        with open(self.testSet, 'rb') as f:
            featureMat = pickle.load(f)
        return featureMat

    def save_data(self):
        print(self.testSet)
        with open(self.testSet, 'wb') as f:
            pickle.dump(self.featureMat, f)

    def plot_data(self,w,b):
        plt.figure(1)
        plt.title('dataSet plot')
        plt.xlabel('x1')
        plt.ylabel('x2')
        colorVect = []
        print(self.featureMat.shape)
        for i in range(len(self.featureMat)):
            if self.featureMat[i, -1] == 1:
                colorVect.append('red')
            else:
                colorVect.append('blue')
        plt.scatter(self.featureMat[:, 0].flatten().A[0], self.featureMat[:, 1].flatten().A[0], color=colorVect, s=15,
                    alpha=0.5)
        x1Vect = np.arange(-1.0, 10.0, 0.1)
        x2Vect = []
        w1 = w[0,0]
        w2 = w[1,0]
        for x in x1Vect:
            x2 = (-b[0,0]-w1*x)/w2
            x2Vect.append(x2)
        plt.plot(x1Vect, x2Vect, 'k-')
        plt.show()

    def select_randJ(self, i, m):
        """
        随机选择一个整数
        Args:
            i  第一个alpha的下标
            m  所有alpha的数目
        Returns:
            j  返回一个不为i的随机数，在0~m之间的整数值
        """
        j = i
        while j == i:
            j = int(random.uniform(0, m))
        return j

    def clipAlpha(self, aj, H, L):
        """clipAlpha(调整aj的值，使aj处于 L<=aj<=H)
        Args:
            aj  目标值
            H   最大值
            L   最小值
        Returns:
            aj  目标值
        """
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj

    def sequential_minimal_optimization(self, C, toler, maxIter):
        """
        无核函数

        Args:
            C   惩罚因子
            toler   松弛因子
            maxIter 退出前最大的循环次数
        Returns:
            b       模型的常量值
            alphas  拉格朗日乘子
        """
        m, n = self.featureMat.shape
        featureData = self.featureMat[:, 0:-1]
        labelData = self.featureMat[:, -1]
        b = 0
        iter = 0
        alphas = np.mat(np.zeros([m, 1]))
        while iter < maxIter:
            alphaPairsChanged = 0
            for i in range(m):
                fxi = np.multiply(alphas, labelData).T * featureData * featureData[i, :].T + b
                Ei = fxi - labelData[i]
                # 判断KNN条件
                if (labelData[i] * Ei < -toler and alphas[i] < C) or (labelData[i] * Ei > toler and alphas[i] > 0):
                    alphaIold = alphas[i].copy()
                    j = self.select_randJ(i, m)
                    alphaJold = alphas[j].copy()
                    fxj = np.multiply(alphas, labelData).T * featureData * featureData[j, :].T + b
                    Ej = fxj - labelData[j]
                    if labelData[i] == labelData[j]:
                        L = max(0, alphas[j] + alphas[i] - C)
                        H = min(C, alphas[j] + alphas[i])
                    else:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(C, alphas[j] - alphas[i] + C)
                    if L == H:
                        print("L==H")
                        continue
                    # eta 反应了i和j之间的距离
                    eta = featureData[i, :] * featureData[i, :].T + featureData[j, :] * featureData[j, :].T - 2.0 * (
                        featureData[i, :] * featureData[j, :].T)
                    if eta <= 0:
                        print("eta<=0")
                        continue
                    alphas[j] += labelData[j] * (Ei - Ej) / eta
                    alphas[j] = self.clipAlpha(alphas[j], H, L)
                    if abs(alphas[j] - alphaJold) < 0.00001:
                        continue
                    alphas[i] += (alphaJold - alphas[j]) * labelData[i] * labelData[j]
                    bi = b - (Ei + labelData[i] * (featureData[i, :] * featureData[i, :].T) * (alphas[i] - alphaIold) + \
                              labelData[j] * (featureData[i, :] * featureData[j, :].T) * (alphas[j] - alphaJold))
                    bj = b - (Ej + labelData[i] * (featureData[i, :] * featureData[i, :].T) * (alphas[i] - alphaIold) + \
                              labelData[j] * (featureData[i, :] * featureData[j, :].T) * (alphas[j] - alphaJold))
                    if (0 < alphas[i]) and (C > alphas[i]):
                        b = bi
                    elif (0 < alphas[j]) and (C > alphas[j]):
                        b = bj
                    else:
                        b = (bi + bj) / 2.0
                    alphaPairsChanged += 1
                    print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
                    # 在for循环外，检查alpha值是否做了更新，如果在更新则将iter设为0后继续运行程序
                    # 知道更新完毕后，iter次循环无变化，才推出循环。
            if (alphaPairsChanged == 0):
                iter += 1
            else:
                iter = 0
            print("iteration number: %d" % iter)
        return b, alphas

    def calcWs(self, alphas, dataArr, classLabels):
        """
        基于alpha计算w值
        Args:
            alphas        拉格朗日乘子
            dataArr       feature数据集
            classLabels   目标变量数据集

        Returns:
            wc  回归系数
        """
        X = np.mat(dataArr)
        labelMat = np.mat(classLabels)
        m, n = X.shape
        w = np.zeros((n, 1))
        for i in range(m):
            w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
        return w


if __name__ == '__main__':
    bsvm = basicSVM()
    b, alphas = bsvm.sequential_minimal_optimization(0.1, 0.01, 10)
    print(alphas)
    w = bsvm.calcWs(alphas,bsvm.featureMat[:, 0:-1],bsvm.featureMat[:, -1])
    bsvm.plot_data(w,b)

