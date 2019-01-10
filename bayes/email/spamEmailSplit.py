#!/usr/bin/python
# -*- coding: UTF-8 -*-

#   data-analysis
#   spamEmailSplit.py

"""
    Description: 垃圾邮件分类（朴素贝叶斯，词带模式）
    author : zhangjingbo 
    Date:    2018/12/11
"""
import ml.bayes.testNativeBayes as nbayes
import os
import pickle
import random
import re
import numpy as np


class spamEmail:
    """
    利用贝叶斯分类，建立垃圾邮件识别机
    """

    def __init__(self):
        self.spamDir = 'spam'
        self.hamDir = 'ham'

    def load_vocab(self):
        """
        从文件加载词汇表
        """
        posting_list = []
        label_list = []
        for file in os.listdir(self.spamDir):
            print(file)
            tmpVect = []
            with open(self.spamDir + '/' + file, 'r', encoding='Windows 1252') as f:
                for str in f:
                    tokenlist = re.split('\W+', str)
                    tmpVect.extend([token.lower() for token in tokenlist if len(token) > 2])
            posting_list.append(list(set(tmpVect)))
            print(tmpVect)
            label_list.append(1)
        for file in os.listdir(self.hamDir):
            print(file)
            tmpVect = []
            with open(self.hamDir + '/' + file, 'r', encoding='Windows 1252') as f:
                for str in f:
                    tokenlist = re.split('\W+', str)
                    tmpVect.extend([token.lower() for token in tokenlist if len(token) > 2])
            posting_list.append(tmpVect)
            print(tmpVect)
            label_list.append(0)
        vocaball = {}
        vocaball['posting_list'] = posting_list
        vocaball['label_list'] = label_list
        with open('vocab_list', 'wb') as f:
            pickle.dump(vocaball, f)

    def test_spam(self):
        with open('vocab_list', 'rb') as f:
            vocaball = pickle.load(f)
        test_set = [int(num) for num in random.sample(range(50), 10)]
        train_set = list(set(range(50)) - set(test_set))
        posting_list = vocaball['posting_list']
        label_list = vocaball['label_list']
        vocab_list = nbayes.create_vocab_list(posting_list)
        trainmat = []
        trainclass = []
        for doc_index in train_set:
            trainmat.append(nbayes.bagofword2vect(vocab_list, posting_list[doc_index]))
            trainclass.append(label_list[doc_index])
        p0, p1, p_class = nbayes.train_native_bayes(np.array(trainmat), np.array(trainclass))
        totalnum = 0
        errnum = 0
        for doc_index in test_set:
            label = nbayes.classify_naive_bayes(nbayes.bagofword2vect(vocab_list, posting_list[doc_index]), p0, p1,
                                                p_class)
            print(posting_list[doc_index])
            print('文档{}的分类为{},贝叶斯分类为{}'.format(doc_index, label_list[doc_index], label))
            if label != label_list[doc_index]:
                errnum += 1
            totalnum += 1
        print(errnum / totalnum)


def main():
    sp = spamEmail()
    sp.test_spam()


if __name__ == '__main__':
    main()
