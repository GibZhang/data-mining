#!/usr/bin/python
# -*- coding: UTF-8 -*-

#   data-analysis
#   MyAdaBossting.py

"""
    Description:
    author : zhangjingbo 
    Date:    2018/12/26
"""
import numpy as np


class AdaBoosting:
    def __init__(self):
        pass

    def file2mat(self, filename):
        numFeat = len(open(filename).readline().split('\t'))
        dataArr = []
        labelArr = []
        fr = open(filename)
        for line in fr.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            for i in range(numFeat - 1):
                lineArr.append(float(curLine[i]))
            dataArr.append(lineArr)
            labelArr.append(float(curLine[-1]))
        return dataArr, labelArr

    def stump_classify(self, datamat, feature_index, thread_value, inequeal):
        """
        (将数据集，按照feature列的value进行 二分法切分比较来赋值分类)
        :param data_mat: Matrix数据集
        :param dimen: 特征的哪一个列
        :param thresh_val: 特征列要比较的值
        :param thresh_ineq:
        :return: np.array
        """
        ret_array = np.ones((datamat.shape[0], 1))
        if inequeal == 'lt':
            ret_array[datamat[:, feature_index] <= thread_value] = -1.0
        else:
            ret_array[datamat[:, feature_index] > thread_value] = -1.0
        return ret_array

    def build_stump(self, data_arr, class_labels, D):
        """
        得到决策树的模型 (这个比较重要，需要看懂）
        :param data_arr: 特征标签集合
        :param class_labels: 分类标签集合
        :param D: 最初的特征权重值
        :return: bestStump    最优的分类器模型
                min_error     错误率
                best_class_est  训练后的结果集
        """
        data_mat = np.mat(data_arr)
        label_mat = np.mat(class_labels).T

        m, n = np.shape(data_mat)
        num_steps = 10
        best_stump = {}
        best_class_est = np.mat(np.zeros((m, 1)))
        # 无穷大
        min_err = np.inf
        for i in range(n):
            feature_max = data_mat[:, i].max()
            feature_min = data_mat[:, i].min()
            step_size = (feature_max - feature_min) / float(num_steps)
            for j in range(num_steps):
                for inequal in ['lt', 'gt']:
                    thread_value = feature_min + j * step_size
                    predicted_values = self.stump_classify(data_mat, i, thread_value, inequal)
                    err_arr = np.ones((m, 1))
                    err_arr[predicted_values == label_mat] = 0.0
                    weighted_err = D.T * err_arr
                    if weighted_err < min_err:
                        min_err = weighted_err
                        best_class_est = predicted_values.copy()
                        best_stump['dim'] = i
                        best_stump['thresh'] = thread_value
                        best_stump['ineq'] = inequal
        return best_stump, min_err, best_class_est

    def ada_boosting(self, dataArr, classArr, numIter):
        weakClassArr = []
        m = np.mat(dataArr).shape[0]
        D = np.mat(np.ones((m, 1)) / m)
        aggClassEst = np.mat(np.zeros((m, 1)))
        for i in range(numIter):
            best_stump, min_err, best_class_est = self.build_stump(dataArr, classArr, D)
            alpha = float(0.5 * np.log((1.0 - min_err) / max(min_err, 1e-16)))
            best_stump['alpha'] = alpha
            weakClassArr.append(best_stump)
            aggClassEst += alpha * best_class_est
            expon = np.multiply(-1 * alpha * np.mat(classArr).T, best_class_est)
            D = np.multiply(D, np.exp(expon))
            D = D / D.sum()
            aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classArr).T, np.ones((m, 1)))
            errorRate = aggErrors.sum() / m
            print(errorRate)
            if errorRate == 0.0:
                break
        return weakClassArr, aggClassEst


if __name__ == '__main__':
    adaboost = AdaBoosting()
    dataSet, labelSet = adaboost.file2mat('horseColicTraining.txt')
    for i in range(len(labelSet)):
        if labelSet[i] == 0:
            labelSet[i] = -1.0
    weakClassArr, aggClassEst = adaboost.ada_boosting(dataSet, labelSet, 30)
    print(aggClassEst)
