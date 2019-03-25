#!/usr/bin/python
# -*- coding: UTF-8 -*-

#   data-analysis
#   BPnetwork.py

"""
    Description: 反向传播神经网络
    author : zhangjingbo 
    Date:    2019/3/16
"""
import numpy as np


class BPneturalNetwork:
    """
    简单的反向传播神经网络
    """

    def __init__(self, inputNodes, hidenNodes, outputNodes, learningRate):
        self.iNodes = inputNodes
        self.hNodes = hidenNodes
        self.oNodes = outputNodes

        # 设置学习率
        self.lr = learningRate

        # 初始化权重
        self.weight_ih = np.random.normal(0.0, pow(hidenNodes, -0.5), (hidenNodes, inputNodes))
        self.weight_ho = np.random.normal(0.0, pow(outputNodes, -0.5), (outputNodes, hidenNodes))

        # 定义激活函数（S函数）
        self.activation_function = lambda x: 1.0 / (1 + np.exp(-x))

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).transpose()
        hiden_inputs = np.dot(self.weight_ih, inputs)
        hiden_outputs = self.activation_function(hiden_inputs)
        final_inputs = np.dot(self.weight_ho, hiden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

    def train(self, inputs_list, outputs_list):
        inputs = np.array(inputs_list, ndmin=2).transpose()
        target = np.array(outputs_list, ndmin=2).transpose()
        hiden_inputs = np.dot(self.weight_ih, inputs)
        hiden_outputs = self.activation_function(hiden_inputs)
        final_inputs = np.dot(self.weight_ho, hiden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_error = target - final_outputs
        hidden_error = np.dot(self.weight_ho.transpose(), output_error)
        self.weight_ho += self.lr * np.dot((output_error * final_outputs * (1 - final_outputs)),
                                           np.transpose(hiden_outputs))
        self.weight_ih += self.lr * np.dot((hidden_error * hiden_outputs * (1 - hiden_outputs)), np.transpose(inputs))


def run():
    pass


if __name__ == '__main__':
    run()
