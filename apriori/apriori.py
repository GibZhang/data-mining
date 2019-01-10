#!/usr/bin/python
# -*- coding: UTF-8 -*-

#   data-analysis
#   apriori.py

"""
    Description: 利用apriori算法发现频繁集
    author : zhangjingbo 
    Date:    2018/12/29
"""
import pickle


def load_data():
    """
    载入数据集
    :return: 数据集
    """
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    """
    构建单项集
    :param dataSet:
    """
    C1 = []
    for tran in dataSet:
        for item in tran:
            if [item] not in C1:
                C1.append([item])
    C1.sort()
    return map(frozenset, C1)


def scanD(D, Ck, minSupport):
    """
    寻找满足支持度的相集
    :param D: 数据集
    :param Ck: 相集
    :param minSupport:最小支持度
    """
    Ck = list(Ck)
    ssCnt = {}  # 记录所有项集的数目
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt.keys():
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = len(D)
    retList = []
    supportedData = {}
    for key in ssCnt:
        support = ssCnt[key] / float(numItems)
        if support >= minSupport:
            retList.insert(0, key)
            supportedData[key] = support
    return retList, supportedData


def aprioriGen(Lk, k):  # 构建一个k项集Ck,eg：要生成一个k=3项集，则可以考虑比较{0,1}{0,2}{1,2} 只需要比较Li[:k-1]即Li[:1]
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


def apriori(dataSet, minSupport):
    """
    生成数据项Ck
    :param dataSet:
    :param minSupport:
    """
    C1 = createC1(dataSet)
    L1, supportData = scanD(dataSet, C1, minSupport)
    L = [L1]
    k = 2
    flag = True
    while flag:
        Ck = aprioriGen(L[k - 2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        if len(Lk) > 0:
            L.append(Lk)
        else:
            flag = False
        k += 1
    return L, supportData


def generateRules(L, supportData, minConf=0.7):
    """
    计算关联规则
    :param L: 频繁项集
    :param supportData: 频繁项集支持度
    :param minConf: 最小可信度
    """
    bigRulesList = []
    for i in range(1, len(L)):  # 选择频繁项中项目集大于1的项
        for freqSet in L[i]:
            H = [frozenset([item]) for item in freqSet]
            # if i > 1:
            rulesFromConseq(freqSet, H, supportData, bigRulesList, minConf)
            # else:
            #     calcConf(freqSet, H, supportData, bigRulesList, minConf)
    return bigRulesList


def calcConf(freqSet, H, supportData, bigRuleList, minConf):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf > minConf:
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            bigRuleList.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqSet, H, supportData, bigRulesList, minConf):
    m = len(H[0])
    if len(freqSet) > m:
        Hmpl = calcConf(freqSet, H, supportData, bigRulesList, minConf)
    if len(freqSet) > m + 1:
        Hmpl = aprioriGen(Hmpl, m + 1)
        if len(Hmpl) > 1:
            rulesFromConseq(freqSet, Hmpl, supportData, bigRulesList, minConf)


def load_mushroom():
    dataSet1 = []
    dataSet2 = []
    with open('mushroom.dat') as f:
        for line in f:
            lineArr = []
            for linestr in line.rstrip(' \n').split(' '):
                lineArr.append(float(linestr))
            firstEle = lineArr.pop(0)
            if firstEle == 1:
                dataSet1.append(lineArr)
            if firstEle == 2:
                dataSet2.append(lineArr)
    return dataSet1, dataSet2


# dataSet1, dataSet2 = load_mushroom()
D = load_data()
L, supportData = apriori(D, 0.5)
brl = generateRules(L, supportData, 0.5)
# print(brl)
