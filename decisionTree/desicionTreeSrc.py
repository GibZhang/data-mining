#!/usr/bin/python
# -*- coding: UTF-8 -*-

#   data-analysis
#   desicionTreeSrc.py

"""
    Description: 决策树算法实现
    author : zhangjingbo 
    Date:    2018/11/30
"""
import math


def createDataset():
    """
     建立数据集
     根据是否可以附上水面，是否有璞 判断是不是鱼
    :return:
    """
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calShannonEnt(dataSet):
    """
    Desc：
        calculate Shannon entropy -- 计算给定数据集的香农熵,Ent(D)
    Args:
        dataSet -- 数据集
    Returns:
        shannonEnt -- 返回 每一组 feature 下的某个分类下，香农熵的信息期望
    """
    # 样本总数
    dataSetNum = len(dataSet)
    labelCounts = {}
    for fetureVect in dataSet:
        currentLabel = fetureVect[-1]
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for count in labelCounts.values():
        shannonEnt += -(count / dataSetNum) * math.log(count / dataSetNum)
    return shannonEnt


def chooseBestFeatureToSplit(dataSet):
    """
    计算dataSet特征的entropy，并选择entro最大的特征作为选择划分数据集的特征
    :param dataSet:
    :return: 最有特征的index
    """
    # 特征维数
    fetureNum = len(dataSet[0]) - 1
    totalNum = len(dataSet)
    baseEntroy = calShannonEnt(dataSet)  # Ent(D)
    # 计算某个特征entropy的公式 ： ent(A) = Ent(D) - sum(Di/D)*Ent(Di)
    bestFetureIndex = 0  # 最佳特征的列值
    bestFetureEntroy = 0.0  # 最佳特征的entropy
    for index in range(fetureNum):
        fetureValue = {}  # 所有的特征值
        for vect in dataSet:
            if vect[index] not in fetureValue:
                fetureValue[vect[index]] = 0
            fetureValue[vect[index]] += 1
        fetureEnt = 0.0
        for key in fetureValue:
            retDataSet = splistDataSet(dataSet, index, key)
            fetureEnt += calShannonEnt(retDataSet) * (fetureValue[key] / totalNum)
        ent = baseEntroy - fetureEnt
        if ent > bestFetureEntroy:
            bestFetureEntroy = ent
            bestFetureIndex = index
    return bestFetureIndex


def splistDataSet(dataSet, index, value):
    """
    将第index列，值为value的数据作为新的数据集返回
    :param dataSet:
    :param index:
    :param value:
    """
    retDataSet = []
    for vect in dataSet:
        if vect[index] == value:
            reducedVet = vect[:index]
            reducedVet.extend(vect[index + 1:])
            retDataSet.append(reducedVet)
    return retDataSet


def majorityLabel(dataSet):
    """
    计算所有label的个数，返回label数最多的label
    :param dataSet:
    """
    labelList = [x[-1] for x in dataSet]
    count = {}
    for label in labelList:
        if label not in count:
            count[label] = 0
        count[label] += 1
    maxNum, target = 0, None
    for k, v in count.items():
        if v > maxNum:
            maxNum = v
            target = k
    return target


def createTree(dataSet, labels):
    """
    建立决策树：结构如图mytree {feature:{{value1:tree1},{value2:tree2},{value3:tree3}}}
    停止条件 1：全部target一致，已经为同一类 2：使用完所有特征，仍然未分类，这是返回多数target
    :return:
    """
    classLabel = [x[-1] for x in dataSet]
    if classLabel.count(classLabel[-1]) == len(classLabel):
        return classLabel[-1]
    if len(dataSet[0]) == 1:
        return majorityLabel(dataSet)

    bestFeatureIndex = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeatureIndex]
    mytree = {bestFeatureLabel: {}}
    # 删除使用过的特征
    del labels[bestFeatureIndex]
    uniqueValue = set([vect[bestFeatureIndex] for vect in dataSet])
    for value in uniqueValue:
        sublabels = labels[:]
        splitedDataSet = splistDataSet(dataSet, bestFeatureIndex, value)
        mytree[bestFeatureLabel][value] = createTree(splitedDataSet, sublabels)
    return mytree


def classifyData(decisionTree, featureLabels, featureVet):
    """
    decisionTree 为createTree获得的决策树，可向下获取最终的target
    :type featureVet list
    :param featureVet: [1,1]
    :type decisionTree dict
    :param decisionTree: {feature:{{value1:tree1},{value2:tree2},{value3:tree3}}}
    :type featureLabels list
    :param featureLabels: ['no surfacing', 'flippers']
    """
    firstFeature = list(decisionTree.keys())[0]
    featureIndex = featureLabels.index(firstFeature)
    subTree = decisionTree[firstFeature][featureVet[featureIndex]]  # 根据输入的feature 获取subtree
    if isinstance(subTree, dict):
        return classifyData(subTree, featureLabels, featureVet)
    return subTree


def load_file(filename):
    dataSet = []
    with open(filename, 'r',encoding='gbk') as f:
        for str in f:
            tmp = str.rstrip('\n').split('\t')
            dataSet.append(tmp)
    return dataSet


def saveDescisonTree(myTree, filename):
    """
    将得到的决策树保存:'fishDesicionTree'
    :type myTree: object
    """
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(myTree, f)


def loadDescionTree(filename):
    """
     读取保存的决策树
    :param filename:
    :return:
    """
    import pickle
    try:
        with open(filename, 'rb') as f:
            mytree = pickle.load(f)
        return mytree
    except FileNotFoundError as e:
        print(e.filename, " not found")


if __name__ == '__main__':
    # dataSet, labelSet = createDataset()
    # print(calShannonEnt(dataSet))
    # print(splistDataSet(dataSet, 0, 1))
    # print(chooseBestFeatureToSplit(dataSet))
    # mytree = createTree(dataSet, labelSet)
    # print(classifyData(mytree, ['no surfacing', 'flippers'], [1, 1]))
    # saveDescisonTree(mytree, 'fishDescison')
    # import ml.decisionTree.desicisonTreePlot as dsplot
    #
    # mytree = loadDescionTree('fishDescison')
    # dsplot.createPlot(mytree)
    # dataSet = load_file('lenses.txt')
    # dataSet = []
    # with open('data.txt','r') as f:
    #     for str in f:
    #         dataSet.append(str.rstrip('\n').split(' '))
    # lensesLabels = ['身高', '体重']
    import ml.decisionTree.desicisonTreePlot as dtp
    #
    # mytree = createTree(dataSet, lensesLabels)
    # saveDescisonTree(mytree, 'dataTree')
    mytree = loadDescionTree('dataTree')
    dtp.createPlot(mytree)
