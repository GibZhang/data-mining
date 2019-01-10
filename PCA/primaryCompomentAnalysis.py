#!/usr/bin/python
# -*- coding: UTF-8 -*-

#   data-analysis
#   primaryCompomentAnalysis.py

"""
    Description:主成分分析，用于数据特征过多，减少弱特征
    author : zhangjingbo 
    Date:    2019/1/5
"""
import requests
import numpy as np


def downloadSecomData():
    data = requests.get('https://raw.githubusercontent.com/apachecn/AiLearning/dev/db/13.PCA/secom.data').content
    with open('secom.data', 'wb') as f:
        f.write(data)


def read_data(filename):
    featArr = []
    with open(filename, 'rb') as f:
        for line in f:
            line = line.decode('utf-8')
            lineArr = []
            for v in line.split(' '):
                lineArr.append(float(v))
            featArr.append(lineArr)
        featMat = np.mat(featArr)
        return featMat


def replaceNanWithMean(dataMat):
    m, n = dataMat.shape
    for i in range(n):
        noNanIndex = np.nonzero(~np.isnan(dataMat[:, i].A))[0]
        mean_value = np.mean(dataMat[noNanIndex, i])
        nanIndex = np.nonzero(np.isnan(dataMat[:, i].A))[0]
        if not len(nanIndex) == 0:
            dataMat[nanIndex, i] = mean_value
    return dataMat


def pca(dataMat, topN=99):
    """
    主成分分析
    :param dataMat: 特征矩阵
    :param topN: 保留99的特征
    :return: featMat 保留主成分后的特征
    """
    meanValues = np.mean(dataMat, axis=0)
    m = dataMat.shape[0]
    x = np.ones((m, 1))
    meanRemoved = dataMat - np.multiply(x, meanValues)
    covMat = np.cov(meanRemoved, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigSum = np.sum(eigVals)
    eigIndex = reversed(np.argsort(eigVals))
    eigSelectedSum = 0
    eigSelectedIndex = []
    for ind in eigIndex:
        eigSelectedSum += eigVals[ind]
        eigSelectedIndex.append(ind)
        if eigSelectedSum * 100 / eigSum >= topN:
            break
    redEigVects = eigVects[:, eigSelectedIndex]
    lowDataMat = dataMat * redEigVects
    return lowDataMat


featMat = read_data('secom.data')
featMat = replaceNanWithMean(featMat)
lowDataMat = pca(featMat)
print(lowDataMat.shape)
