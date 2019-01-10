#!/usr/bin/python
# -*- coding: UTF-8 -*-

#   data-analysis
#   TreeExplore.py

"""
    Description: 可视化探索数据，数回归
    author : zhangjingbo 
    Date:    2018/12/27
"""
import matplotlib.pylab as plt
import numpy as np


def file2Arr(filename='ex00.txt'):
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


def arrPlot(dataArr, yHat=None):
    """
    画出数据点，观察数据特征
    :param dataArr: 数组数据
    """
    plt.figure(1)
    plt.xlabel('x1')
    plt.ylabel('x2')
    dataMat = np.mat(dataArr)
    plt.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0])
    plt.scatter(dataMat[:, 0].flatten().A[0],yHat.flatten().A[0],color='red')
    plt.show()


def binSplitDataSet(dataMat, feature, value):
    """
    数据集按特征列，对应value分为两类
    :param dataSet:
    :param feature:
    :param value:
    """
    n = dataMat.shape[1]
    left_index = np.nonzero(dataMat[:, feature] <= value)[0]
    right_index = np.nonzero(dataMat[:, feature] > value)[0]
    if len(left_index) != 0:
        left = dataMat[left_index, :]
    else:
        left = np.zeros((1, n))
    if len(right_index) != 0:
        right = dataMat[right_index, :]
    else:
        right = np.zeros((1, n))
    return left, right


def regLeaf(dataSet):  # returns the value used for each leaf 返回叶节点的平均值，最为回归值
    return np.mean(dataSet[:, -1])


def regErr(dataSet):  # 返回数据集的方差
    return np.var(dataSet[:, -1]) * dataSet.shape[0]


# 线性拟合
def linearSolve(dataSet):  # helper function used in two places
    m, n = dataSet.shape
    X = np.mat(np.ones((m, n)))
    Y = np.mat(np.ones((m, 1)))  # create a copy of data with 1 in 0th postion
    X[:, 1:n] = dataSet[:, 0:n - 1]
    Y = dataSet[:, -1]  # and strip out Y
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def modelLeaf(dataSet):  # create linear model and return coeficients
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(np.power(Y - yHat, 2))


def chooseBestFeature(dataSet, leafType, leafErr, tols):
    """
    选择最优的属性划分数据集
    :param leafType: 叶子节点的value
    :param leafErr: 叶子节点的方差
    :param tols: 两个参数，停止条件 1.数据集二分后，方差减小小于tols[0]，2.叶子节点数据量小于 tols[1]
    :param dataSet: 数据集
    :return: feature_index,feature_value
    """
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    bestFeatureIndex = -1
    bestFeatureValue = None
    dataSetVar = leafErr(dataSet)
    bestVar = dataSetVar
    m, n = dataSet.shape
    for feature in range(n - 1):
        for featureValue in set(dataSet[:, feature].T.tolist()[0]):
            left, right = binSplitDataSet(dataSet, feature, featureValue)
            if left.shape[0] == 1 or right.shape[0] == 1:
                continue
            newVar = leafErr(left) + leafErr(right)
            if newVar < bestVar:
                bestVar = newVar
                bestFeatureIndex = feature
                bestFeatureValue = featureValue
    if dataSetVar - bestVar < tols[0]:  # 方差减小太少
        return None, leafType(dataSet)
    left, right = binSplitDataSet(dataSet, bestFeatureIndex, bestFeatureValue)
    if left.shape[0] < tols[1] or right.shape[0] < tols[1]:
        return None, leafType(dataSet)
    return bestFeatureIndex, bestFeatureValue


def createTree(dataSet, leafType, errType, tols):
    """
    创建回归树
    :param dataSet:
    """
    feat, value = chooseBestFeature(dataSet, leafType, errType, tols)
    if feat == None:
        return value
    tree = {}
    tree['feat'] = feat
    tree['value'] = value
    left_tree, right_tree = binSplitDataSet(dataSet, feat, value)
    tree['left'] = createTree(left_tree, leafType, errType, tols)
    tree['right'] = createTree(right_tree, leafType, errType, tols)
    return tree


def isTree(obj):
    return (type(obj).__name__ == 'dict')


def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, testData):
    """
    剪枝
    :param tree:
    :param testData:
    :return:
    """
    if testData.shape[0] == 0:
        return getMean(tree)
    leftDataSet, rightDataSet = binSplitDataSet(testData, tree['feat'], tree['value'])
    if isTree(tree['left']) or isTree(tree['right']):  # 如果有一个是树状结构，就继续深入剪枝
        if isTree(tree['left']):
            tree['left'] = prune(tree['left'], leftDataSet)  # 返回剪枝后的左树
        if isTree(tree['right']):
            tree['right'] = prune(tree['right'], rightDataSet)  # 返回剪枝后的右树
        return tree
    else:
        """
        left和right 都不是树，则检查是否可以合并左边和右边
        """
        errorNomerge = np.sum(np.power(leftDataSet[:, -1] - tree['left'], 2)) + np.sum( np.power(rightDataSet[:, -1] - tree['right'], 2))
        dataSetMean = np.mean(testData[:, -1])
        errorMerge = np.sum(np.power(testData[:, -1] - dataSetMean, 2))
        if errorMerge < errorNomerge:
            print('Merge')
            return dataSetMean
        else:
            return tree


"""
回归预测
"""


def regTreeEval(model, inDat):
    return float(model)  # 直接返回值


def modelTreeEval(model, inDat):  # 计算拟合值
    n = inDat.shape[1]
    X = np.mat(np.ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['feat']] <= tree['value']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat


dataArr = file2Arr('bikeSpeedVsIq_train.txt')
dataSet = np.mat(dataArr)
tree = createTree(dataSet, regLeaf, regErr, (1, 20))
dataList = []
for point in dataArr:
    dataList.append(point[0])
yHat = createForeCast(tree, dataList, regTreeEval)
print(np.corrcoef(dataList,yHat.T[0].tolist()[0]))
arrPlot(dataArr, yHat)

# dataArr = file2Arr('expTest.txt')
# dataSet = np.mat(dataArr)
# tree_pruned = prune(tree, dataSet)
# print(tree_pruned)
