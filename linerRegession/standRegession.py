# coding=utf-8
import pandas as pd
import matplotlib.pylab as plt
import numpy as np


def load_data_file(filename):
    """
    从数据文件载入数据
    :param filename: 数据文件名称 dir+filename
    :return:  dataMat=[] 参数矩阵 labelMat=[] 特征矩阵
    """
    dataMat = []
    labelMat = []
    with open(filename) as f:
        while True:
            l = f.readline()
            if not l:
                break
            strs = l.split("\t")
            tmp = []
            for i in range(len(strs)-1):
                tmp.append(float(strs[i]))
            dataMat.append(tmp)
            labelMat.append(float(strs[-1].split('\n')[0]))
    return dataMat,labelMat


def stand_liner_regession(dataMat, labelMat):
    """
    Description:
    标准线性回归算法求解Q
    :param dataMat: 输入的样本数据，包含每个样本数据的 feature
    :param labelMat: 对应于输入数据的类别标签，也就是每个样本对应的目标变量
    :return: qMat 回归系数
    """
    xMat = np.mat(dataMat)
    yMat = np.mat(labelMat).T
    xTx = (xMat.T).dot(xMat)
    if np.linalg.det(xTx) == 0.0:
        print('行列式为0，不可求逆')
        return
    qMat = xTx.I * (xMat.T * yMat)
    return qMat


def regession_plt(filename):
    """
    Description:
    读取数据文件，进行线性回归，并画出数据点和拟合直线
    :param filename: 数据文件名
    :return: None
    """
    dm,lm = load_data_file(filename)
    qMat = stand_liner_regession(dm,lm)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.set_xlabel('x-value')
    ax.set_ylabel('y-value')
    ax.set_title('stand liner regession')
    ax.scatter([np.mat(dm)[:,1].flatten()],[np.mat(lm).T.flatten()],color='r',linewidths=0.8)
    xcopy = np.mat(dm).copy()
    xcopy.sort(0)
    yasum = xcopy.dot(qMat)
    ax.plot(xcopy[:,1],yasum,color='b')
    plt.show()




if __name__ == '__main__':
    regession_plt('data.txt')

