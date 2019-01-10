#!/usr/bin/python
# -*- coding: UTF-8 -*-

#   data-analysis：logisticRegeession
#   baseLogistic.py

"""
    Description: 逻辑回归对testSet进行分类
    author : zhangjingbo 
    Date:    2018/12/18
"""
import os
import pickle
import ml.logisticRegression.basicSetPlot as lr
import numpy as np
import math


class basicLogisticRegression:
    """
    利用最基本的逻辑回归，对testSet进行学习，将数据分类
    """

    def __init__(self, dataFile='testSet'):
        self.testSetFile = dataFile + '.txt'
        self.dataFile = dataFile
        if not os.path.isfile(self.dataFile):
            dataMat = self.read_file()
            self.save_data(dataMat)
        self.load_data()

    def read_file(self):
        try:
            dataMat = []
            with open(self.testSetFile) as f:
                for data_str in f:
                    tmp = ['1.0']
                    tmp.extend(data_str.rstrip('\n').split('\t'))
                    dataMat.append(tmp)
            return np.array(dataMat).astype(float)
        except FileNotFoundError as e:
            print(e.filename, "not found, please check filename first!")

    def save_data(self, dataMat):
        with open(self.dataFile, 'wb') as f:
            pickle.dump(dataMat, f)

    def load_data(self):
        with open(self.dataFile, 'rb') as f:
            dataMat = pickle.load(f)
            self.dataMat = dataMat

    def logistic_regression(self, xinput):
        xVect = [0.0]
        xVect.extend(xinput)
        xVect = np.mat(xVect)
        inx = xVect * self.thetaMat
        yhat = self.sigmod(inx)
        if yhat > 0.5:
            return 1.0
        elif yhat < 0.5:
            return 0.0
        else:
            return None

    def sigmod(self, inX):
        return 1.0 / (1.0 + np.exp(-inX))

    def gradAscd(self, alpha=0.01):
        m = self.dataMat.shape[1]
        thetaMat = np.ones((m - 1, 1))
        featureMat = np.mat(self.dataMat[:, 0:m - 1])
        labelMat = np.mat(self.dataMat[:, -1])
        maxCicle = 500
        for i in range(maxCicle):
            for j in range(featureMat.shape[0]):
                inX = featureMat[j] * thetaMat
                yhat = self.sigmod(inX)
                err = labelMat.T[j] - yhat
                thetaMat += alpha * featureMat[j].T * err
        self.thetaMat = thetaMat
        return thetaMat

    def err_test(self):
        totalNum = 0.0
        errNum = 0.0
        for data in self.dataMat:
            totalNum += 1
            inx = 0.0
            for i in range(self.thetaMat.shape[0]):
                inx += self.thetaMat[i] * data[i]
            yhat = self.sigmod(inx)
            if yhat > 0.5:
                yhat = 1.0
            else:
                yhat = 0.0
            if yhat != data[-1]:
                errNum += 1
        print('error percentage is %2.2f' % (errNum / totalNum))


if __name__ == '__main__':
    blg = basicLogisticRegression('horseColicTraining')
    blg.gradAscd()
    dataMat = blg.dataMat
    thetaMat = blg.thetaMat
    blg.err_test()
    # result = blg.logistic_regression([0.0, 0.0])
    # print(result)
    # ldp = lr.logistic_data_plot(dataMat, thetaMat)
    # ldp.data_plot()
