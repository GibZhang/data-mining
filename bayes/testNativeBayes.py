#!/usr/bin/python
# -*- coding: UTF-8 -*-

#   data-analysis
#   testNativeBayes.py

"""
    Description: 朴素贝叶斯
    author : zhangjingbo 
    Date:    2018/12/10
"""
import numpy as np


def load_data_set():
    """
    创建数据集,都是假的 fake data set
    :return: 单词列表posting_list, 所属类别class_vec
    """
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'gar e'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 is 侮辱性的文字, 0 is not
    return posting_list, class_vec


def create_vocab_list(dataset):
    """
    获取所有单词的集合
    :param data_set: 数据集
    :return: 所有单词的集合(即不含重复元素的单词列表)
    """
    vocab_set = set()
    for item in dataset:
        vocab_set |= set(item)
    return list(vocab_set)


def word2vect(vocab_list, inputset):
    """
    朴素贝叶斯分类原版
    :param train_mat: 文件单词类型矩阵
                    总的输入文本，大致是 [[0,1,0,1,1...], [], []]
    :param train_category: 文件对应的类别分类， [0, 1, 0],
                         列表长度等于单词矩阵数，其中的1代表对应的文件是侮辱性文件，0代表不是侮辱性矩阵
    :return:
    """
    result = [0] * len(vocab_list)
    for word in inputset:
        if word in vocab_list:
            result[vocab_list.index(word)] = 1
    return result


def bagofword2vect(vocab_list, inputset):
    """
    朴素贝叶斯分类原版
    :param train_mat: 文件单词类型矩阵
                    总的输入文本，大致是 [[0,1,0,1,1...], [], []]
    :param train_category: 文件对应的类别分类， [0, 1, 0],
                         列表长度等于单词矩阵数，其中的1代表对应的文件是侮辱性文件，0代表不是侮辱性矩阵
    :return:
    """
    result = [0] * len(vocab_list)
    for word in inputset:
        if word in vocab_list:
            result[vocab_list.index(word)] += 1
    return result


def train_native_bayes(trainmat, train_catory):
    """
    朴素贝叶斯分类修正版，　注意和原来的对比，为什么这么做可以查看书
    :param train_mat:  type is ndarray
                    总的输入文本，大致是 [[0,1,0,1], [], []]
    :param train_category: 文件对应的类别分类， [0, 1, 0],
                            列表的长度应该等于上面那个输入文本的长度
    :return:
    """
    doc_num = len(trainmat)
    word_num = len(trainmat[0])
    pc1 = np.sum(train_catory) / doc_num

    p0num = np.ones(word_num)
    p0numall = 2.0
    p1num = np.ones(word_num)
    p1numall = 2.0

    for i in range(doc_num):
        if train_catory[i] == 0:
            p0num += trainmat[i]
            p0numall += 1
        else:
            p1num += trainmat[i]
            p1numall += 1
    return np.log(p0num / p0numall), np.log(p1num / p1numall), pc1


def classify_naive_bayes(vec2classify, p0vec, p1vec, p_class):
    """
    使用算法：
        # 将乘法转换为加法
        乘法：P(C|F1F2...Fn) = P(F1F2...Fn|C)P(C)/P(F1F2...Fn)
        加法：P(F1|C)*P(F2|C)....P(Fn|C)P(C) -> log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    :param vec2classify: 待测数据[0,1,1,1,1...]，即要分类的向量
    :param p0vec: 类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    :param p1vec: 类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
    :param p_class1: 类别1，侮辱性文件的出现概率
    :return: 类别1 or 0
    """
    p1 = np.sum(vec2classify * p1vec) + np.log(p_class)
    p0 = np.sum(vec2classify * p0vec) + np.log(1 - p_class)
    if p1 > p0:
        return 1
    else:
        return 0


def getclass():
    """
    测试朴素贝叶斯算法
    :return: no return
    """
    # 1. 加载数据集
    list_post, list_classes = load_data_set()
    # 2. 创建单词集合
    vocab_list = create_vocab_list(list_post)
    # 3. 计算单词是否出现并创建数据矩阵
    train_mat = []
    for post_in in list_post:
        train_mat.append(
            # 返回m*len(vocab_list)的矩阵， 记录的都是0，1信息
            # 其实就是那个东西的句子向量（就是data_set里面每一行,也不算句子吧)
            word2vect(vocab_list, post_in)
        )
    p0, p1, p_class = train_native_bayes(np.array(train_mat), np.array(list_classes))

    # 5. 测试数据
    test_one = ['love', 'my', 'dalmation']
    test_one_doc = np.array(word2vect(vocab_list, test_one))
    print('the result is: {}'.format(classify_naive_bayes(test_one_doc, p0, p1, p_class)))
    test_two = ['stupid', 'garbage']
    test_two_doc = np.array(word2vect(vocab_list, test_two))
    print('the result is: {}'.format(classify_naive_bayes(test_two_doc, p0, p1, p_class)))


def load_vocab_list():
    for i in range(1, 26):
        file = 'ham/' + str(i) + '.txt'
        print(file)


def spam_email_split():
    pass


if __name__ == '__main__':
    load_vocab_list()
