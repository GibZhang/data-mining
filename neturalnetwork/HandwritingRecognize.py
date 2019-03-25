#!/usr/bin/python
# -*- coding: UTF-8 -*-

#   data-analysis
#   HandwritingRecognize.py

"""
    Description: 手写数字识别
    author : zhangjingbo 
    Date:    2019/3/17
"""
from ml.neturalnetwork.BPnetwork import BPneturalNetwork
from ml.neturalnetwork.mnist import *
import numpy as np

inputNodes = 28 * 28
outputNodes = 10
hidenNodes = 200

learningRate = 0.1
neturalNetwork = BPneturalNetwork(inputNodes, hidenNodes, outputNodes, learningRate)

trainSet, trainLabel, testSet, testLabel = run()
trainSize = len(trainLabel)
for k in range(5):
    for i in range(trainSize):
        image = trainSet[i]
        label = trainLabel[i]
        inputs = []
        for l in image:
            inputs.extend(l)
        input_list = [x / 255.0 * 0.99 + 0.01 for x in inputs]
        output_list = [0.01] * 10
        output_list[int(label)] = 0.99
        neturalNetwork.train(input_list, output_list)
testPredicts = []
i = 0
for image in testSet:
    inputs = []
    for l in image:
        inputs.extend(l)
    input_list = [x / 255.0 * 0.99 + 0.01 for x in inputs]
    pred = np.argmax(neturalNetwork.query(input_list))
    i += 1
    testPredicts.append(pred)
print(np.array(testPredicts).sum() / len(testLabel))
