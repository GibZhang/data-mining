#!/usr/bin/python
# -*- coding: UTF-8 -*-

#   data-analysis
#   myRandomForest.py

"""
    Description:
    author : zhangjingbo 
    Date:    2018/12/26
"""

import random


class MyRandomRorest:
    def file2matrix(self, filename='sonar_all_data.txt'):
        """
        读取文件
        :param filename:
        """
        dataSet = []
        with open(filename, 'r') as f:
            for str in f:
                lineArr = []
                for lineStr in str.rstrip('\n').split(','):
                    if lineStr.isalpha():
                        lineArr.append(lineStr)
                    else:
                        lineArr.append(float(lineStr))
                dataSet.append(lineArr)
        return dataSet

    def subsample(self, dataSet, ratio):
        """random_forest(评估算法性能，返回模型得分)
        有放回抽样，用于随机选择训练集
        Args:
            dataset         训练数据集
            ratio           训练数据集的样本比例
        Returns:
            sample          随机抽样的训练样本
        """
        sample = []
        n_sample = round(ratio * len(dataSet))
        while len(sample) < n_sample:
            index = random.randrange(len(dataSet))
            sample.append(dataSet[index])
        return sample

    def cross_validation_split(self, dataset, n_folds):
        """
        用于交叉验证
        :param dataset:
        :param n_folds:
        :return:
        """
        dataset_split = []
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for i in range(n_folds):
            fold = []
            while len(fold) < fold_size:
                index = random.randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split

    def build_tree(self, train, max_depth, min_size, n_features):
        root = self.get_best_feature(train, n_features)
        self.split(root, max_depth, min_size, n_features, 1)
        return root

    def split(self, node, max_depth, min_size, n_features, depth):
        left, right = node['groups']
        del node['groups']
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        # check for max depth
        if depth >= max_depth:  # max_depth=10 表示递归十次，若分类还未结束，则选取数据中分类标签较多的作为结果，使分类提前结束，防止过拟合
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        # process left child
        if len(left) <= min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_best_feature(left,
                                                 n_features)  # node['left']是一个字典，形式为{'index':b_index, 'value':b_value, 'groups':b_groups}，所以node是一个多层字典
            self.split(node['left'], max_depth, min_size, n_features, depth + 1)  # 递归，depth+1计算递归层数
        # process right child
        if len(right) <= min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_best_feature(right, n_features)
            self.split(node['right'], max_depth, min_size, n_features, depth + 1)

    # Create a terminal node value # 输出group中出现次数较多的标签
    def to_terminal(self, group):
        outcomes = [row[-1] for row in group]  # max() 函数中，当 key 参数不为空时，就以 key 的函数对象为判断的标准
        return max(set(outcomes), key=outcomes.count)  # 输出 group 中出现次数较多的标签

    def gini(self, groups, class_values):
        gini = 0.0
        for class_value in class_values:  # class_values = [0, 1]
            for group in groups:  # groups = (left, right)
                size = len(group)
                if size == 0:
                    continue
                proportion = ([row[-1] for row in group].count(class_value)) / float(size)
                gini += (proportion * (1.0 - proportion))  # 个人理解：计算代价，分类越准确，则 gini 越小
        return gini

    def get_best_feature(self, train, n_features):
        feature_index = []
        m = len(train)
        n = len(train[0])
        while len(feature_index) < n_features:
            index = random.randrange(n - 1)
            if index not in feature_index:
                feature_index.append(index)  # 属性扰动
        best_index, best_value, best_score, best_groups = 999, 999, 999, None
        class_values = list(set([value[-1] for value in train]))
        for index in feature_index:
            for row in range(m):
                value = train[row][index]
                groups = self.test_split(index, value, train)
                gini_index = self.gini(groups, class_values)
                if gini_index < best_score:
                    best_index, best_value, best_score, best_groups = index, value, gini_index, groups
        return {"b_index": best_index, "b_value": best_value, "groups": best_groups}

    def test_split(self, index, value, train):
        left = []
        right = []
        m = len(train)
        for i in range(m):
            tmp = train[i][index]
            if tmp < value:
                left.append(train[i])
            else:
                right.append(train[i])
        return left, right

    # Make a prediction with a decision tree
    def predict(self, node, row):  # 预测模型分类结果
        if row[node['b_index']] < node['b_value']:
            if isinstance(node['left'], dict):  # isinstance 是 Python 中的一个内建函数。是用来判断一个对象是否是一个已知的类型。
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']

    def bagging_predict(self, trees, row):
        predictions = [self.predict(tree, row) for tree in trees]
        return max(set(predictions), key=predictions.count)

    def random_forest(self, train, test, max_depth, min_size, sample_size, n_trees, n_features):
        trees = []
        for i in range(n_trees):
            # 随机抽样的训练样本， 随机采样保证了每棵决策树训练集的差异性
            sample = self.subsample(train, sample_size)
            # 创建一个决策树
            tree = self.build_tree(sample, max_depth, min_size, n_features)
            trees.append(tree)

        # 每一行的预测结果，bagging 预测最后的分类结果
        predictions = [self.bagging_predict(trees, row) for row in test]
        return predictions

    def accuracy_metric(self, actual_values, predict_values):
        correct = 0
        for i in range(len(actual_values)):
            if actual_values[i] == predict_values[i]:
                correct += 1
        return correct / float(len(actual_values)) * 100.0

    # 评估算法性能，返回模型得分
    def evaluate_algorithm(self, dataset, algorithm, n_folds, *args):
        folds = self.cross_validation_split(dataset, n_folds)
        scores = []
        for fold in folds:
            train = list(folds)
            train.remove(fold)
            train = sum(train, [])
            actual_values = []
            for row in fold:
                actual_values.append(row[-1])
            predicted = algorithm(train, fold, *args)
            score = self.accuracy_metric(actual_values, predicted)
            scores.append(score)
        return scores


if __name__ == '__main__':
    rf = MyRandomRorest()
    # 加载数据
    dataset = rf.file2matrix()  # print(dataset)

    n_folds = 5  # 分成5份数据，进行交叉验证
    max_depth = 20  # 调参（自己修改） #决策树深度不能太深，不然容易导致过拟合
    min_size = 1  # 决策树的叶子节点最少的元素数量
    sample_size = 1.0  # 做决策树时候的样本的比例
    # n_features = int((len(dataset[0])-1))
    n_features = 15  # 调参（自己修改） #准确性与多样性之间的权衡
    for n_trees in [1, 10, 20]:  # 理论上树是越多越好
        scores = rf.evaluate_algorithm(dataset, rf.random_forest, n_folds, max_depth, min_size, sample_size, n_trees,
                                    n_features)
        # 每一次执行本文件时都能产生同一个随机数
        random.seed(1)
        print('random=', random.random())
        print('Trees: %d' % n_trees)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
