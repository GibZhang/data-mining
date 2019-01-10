#!/usr/bin/python
# -*- coding: UTF-8 -*-

#   data-analysis
#   basicSetPlot.py

"""
    Description: testSet画图
    author : zhangjingbo 
    Date:    2018/12/18
"""
import matplotlib.pylab as plt
import numpy as np


class logistic_data_plot:
    def __init__(self, dataMat, thetaMat):
        self.thetaMat = thetaMat
        self.dataMat = dataMat

    def data_plot(self):
        plt.figure(1)
        plt.title('dataSet plot')
        plt.xlabel('x1')
        plt.ylabel('x2')
        colorVect = []
        for i in range(len(self.dataMat)):
            if self.dataMat[i][-1] == 1:
                colorVect.append('red')
            else:
                colorVect.append('blue')
        plt.scatter(self.dataMat[:, 1].flatten(), self.dataMat[:, 2].flatten(), color=colorVect, s=15, alpha=0.5)
        x1Vect = np.arange(-3.0, 3.0, 0.1)
        x2Vect = []
        for x in x1Vect:
            x2 = (-self.thetaMat[0] - x * self.thetaMat[1]) / self.thetaMat[2]
            x2Vect.append(x2)
        x2Vect = np.array(x2Vect)
        plt.plot(x1Vect,x2Vect,'k-')
        plt.show()
