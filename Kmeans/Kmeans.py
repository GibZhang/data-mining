#!/usr/bin/python
# -*- coding: UTF-8 -*-

#   data-analysis
#   Kmeans.py

"""
    Description: K均值做聚类算法
    author : zhangjingbo 
    Date:    2018/12/27
"""
import matplotlib.pylab as plt
import numpy as np
import random
import math


def file2Arr(filename='testSet.txt'):
    """
    读取文件内容，转化为arr
    """
    with open(filename, 'r') as f:
        dataArr = []
        for line in f:
            lineArr = []
            for lineStr in line.rstrip('\n').split('\t'):
                lineArr.append(float(lineStr))
            dataArr.append(lineArr)
    return dataArr


def arrPlot(dataArr, labelVect, K):
    """
    画出数据点，观察数据特征
    :param dataArr: 数组数据
    """
    plt.figure(1)
    plt.xlabel('x1')
    plt.ylabel('x2')
    dataMat = np.mat(dataArr)
    colors = ['b', 'g', 'r', 'orange']
    colorVect = []
    for value in labelVect:
        colorVect.append(colors[int(value)])
    plt.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], c=colorVect)
    plt.show()


def distEclud(vecA, vecB):
    """
    计算距离
    :type vecA:
    """
    return np.sqrt(np.power(vecA - vecB, 2).sum())  # la.norm(vecA-vecB)


def randCent(dataArr, k):
    dataSet = np.mat(dataArr)
    n = dataSet.shape[1]
    centroids = np.mat(np.zeros((k, n)))  # create centroid mat
    for j in range(n):  # create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        for l in range(k):
            centroids[l, j] = np.mat(minJ + rangeJ * random.random())
    return centroids


def kmeansAlgthm(dataArr, muArr):
    """
    原型聚类算法
    :param dataArr: 数据集
    :param muArr: 随机中心
    """
    dataSet = np.mat(dataArr)
    m, n = dataSet.shape
    K = muArr.shape[0]
    clusterAssment = np.zeros((m, 2))
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            vectA = dataSet[i, :]
            minDist = math.inf
            minIndex = -1
            for j in range(K):
                vectB = muArr[j, :]
                dist = distEclud(vectA, vectB)
                if dist < minDist:
                    minDist = dist
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist
        for cent in range(K):
            K_cluster_points = dataSet[np.nonzero(clusterAssment[:, 0] == cent)[0]]
            muArr[cent] = np.mean(K_cluster_points, axis=0)
    return muArr, clusterAssment


K = 3
dataArr = file2Arr('testSet2.txt')
cens = randCent(dataArr, K)
muArr, clusterAssment = kmeansAlgthm(dataArr, cens)
print(muArr)
labelVect = clusterAssment[:, 0]
arrPlot(dataArr, labelVect, K)
