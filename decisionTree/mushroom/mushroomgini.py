#!/usr/bin/python
# -*- coding: UTF-8 -*-

#   data-analysis
#   mushroomgini.py

"""
    Description: 使用gini指数建立决策树，判断蘑菇是否有毒
    author : zhangjingbo 
    Date:    2018/12/11
"""
import pandas as pd
import pickle


class mushroom:
    def __init__(self, mushroomfile, filename, treefile='mushroom.tree', testRatio=0.1):
        self.filename = filename
        self.testRatio = testRatio
        self.mushroomfile = mushroomfile
        self.treefile = treefile

    def file2dataframe(self):
        df = pd.read_csv(self.mushroomfile)
        targetVect = df['class']
        featureVect = df.drop(['class'], axis=1)
        offset = df.shape[0] - int(self.testRatio * df.shape[0])
        trainFeature = featureVect.iloc[:offset]
        trainClass = targetVect.iloc[:offset]
        testFeature = featureVect.iloc[offset:]
        testClass = targetVect.iloc[offset:]
        mushroomDict = {}
        mushroomDict['trainFeature'] = trainFeature
        mushroomDict['trainClass'] = trainClass
        mushroomDict['testFeature'] = testFeature
        mushroomDict['testClass'] = testClass
        with open(self.filename, 'wb') as f:
            pickle.dump(mushroomDict, f)

    def load_data(self):
        mushroomDict = {}
        try:
            with open(self.filename, 'rb') as f:
                mushroomDict = pickle.load(f)
        except FileNotFoundError as e:
            print(e.filename, ' not found !')
            exit(1)
        trainFeature = mushroomDict['trainFeature']
        trainClass = mushroomDict['trainClass']
        return trainFeature, trainClass

    def gini(self, trainClass):
        totalnum = trainClass.shape[0]
        value_counts = trainClass.value_counts()
        ptotal = 0.0
        for countnum in value_counts:
            p = (countnum / totalnum)
            ptotal += p ** 2
        return 1 - ptotal

    def gini_index(self, trainFeature, featurelabel, trainClass):
        gini_index = 0.0
        totalD = trainFeature.shape[0]
        for item in trainFeature[featurelabel].value_counts().items():
            sub_trainclass = trainClass[trainFeature[featurelabel] == item[0]]
            gini_index += (item[1] / totalD) * self.gini(sub_trainclass)
        return gini_index

    def chooseBestFeatureToSplit(self, trainFeature, trainClass):
        minGini = None
        bestlabel = None
        for featurelabel in trainFeature.columns:
            tmp = self.gini_index(trainFeature, featurelabel, trainClass)
            if minGini is None:
                minGini = tmp
                bestlabel = featurelabel
            if tmp < minGini:
                minGini = tmp
                bestlabel = featurelabel
        return bestlabel

    def splistDataSet(self, dataSet, dataClass, label, value):
        subDataSet = dataSet[dataSet[label] == value]
        subDataClass = dataClass[dataSet[label] == value]
        subDataSet = subDataSet.drop(labels=[label], axis=1)
        return subDataSet, subDataClass

    def majorityLabel(self, dataClass):
        tmp = dataClass.value_counts(sort=True, ascending=False)
        return tmp.index[0]

    def createTree(self, dataSet, dataClass):
        if dataClass.unique().shape[0] == 1:
            return dataClass.unique()[0]
        if dataSet.shape[1] == 1:
            return self.majorityLabel(dataClass)
        bestFeatureLabel = self.chooseBestFeatureToSplit(dataSet, dataClass)
        mytree = {bestFeatureLabel: {}}
        for value in dataSet[bestFeatureLabel].unique():
            splitedDataSet, splitedDataClass = self.splistDataSet(dataSet, dataClass, bestFeatureLabel, value)
            mytree[bestFeatureLabel][value] = self.createTree(splitedDataSet, splitedDataClass)
        return mytree

    def saveTree(self, mytree):
        with open(self.treefile, 'wb') as f:
            pickle.dump(mytree, f)

    def classifyData(self, decisionTree, features):
        firstFeature = list(decisionTree.keys())[0]
        subTree = decisionTree[firstFeature][features[firstFeature]]
        if isinstance(subTree, dict):
            return self.classifyData(subTree, features)
        return subTree

    def testData(self, decisionTree, featureLabels, classLabels):
        errnum = 0
        totalnum = featureLabels.shape[0]
        for i in range(totalnum):
            classifiedLabel = self.classifyData(decisionTree, featureLabels.iloc[i])
            if classifiedLabel != classLabels.iloc[i]:
                errnum += 1
        return errnum / totalnum


def main():
    # mr = mushroom('mushrooms_new.csv', 'mr')
    # mr.file2dataframe()
    # fm, lm = mr.load_data()
    # mr.chooseBestFeatureToSplit(fm, lm)
    # mr.majorityLabel(lm)
    # mytree = mr.createTree(fm, lm)
    # mr.saveTree(mytree)
    # mushroomDict = {}
    # try:
    #     with open('mr', 'rb') as f:
    #         mushroomDict = pickle.load(f)
    # except FileNotFoundError as e:
    #     print(e.filename, ' not found !')
    #     exit(1)
    # testFeature = mushroomDict['testFeature']
    # testClass = mushroomDict['testClass']
    with open('mushroom.tree', 'rb') as f:
        mytree = pickle.load(f)
    print(mytree)
    import ml.decisionTree.desicisonTreePlot as dpl
    dpl.createPlot(mytree)
    # e = mr.testData(mytree, testFeature, testClass)
    # print('error percatage is {}'.format(e))


if __name__ == '__main__':
    main()
