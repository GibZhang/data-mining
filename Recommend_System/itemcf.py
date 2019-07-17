# -*- coding: utf-8 -*-
# !/usr/bin/env python
# coding=utf-8

"""
	@author:	xin.jin
	@file:		item_based_cf.py
	@time:		2019.07.02
	@notes:		
	
"""

import numpy as np
import pandas as pd
import os
import sys
import warnings

from operator import itemgetter
from collections import defaultdict
from sklearn.model_selection import train_test_split

random_seed = 0
warnings.filterwarnings("ignore")


class ItemCf(object):
	"""
	item based collaborative filtering
	"""

	def __init__(self):
		self.seed = 0
		self.data_path = r"E:\MyProgramming\python_workspace\55. recommendation\my rcmd\data_movieLens" \
                         + os.sep + "ml-1m" + os.sep
		# 总数据
		self.ratings = None           # 用户商品评分表（总）, df
		self.n_users = None           # 用户数（总）
		self.n_items = None           # 商品数（总）

		# 训练集，测试集数据
		self.train = None             # 用户商品评分表（训练集）, df
		self.test = None              # 用户商品评分表（测试集）, df
		self.train_dict = dict()      # 用户-商品倒排表（训练集）, dict, {user: {item: rating, ...}, ...}
		self.test_dict = dict()       # 用户-商品倒排表（测试集）, dict, {user: {item: rating, ...}, ...}
		self.n_users_train = None     # 用户数（训练集）
		self.n_items_train = None     # 商品数（训练集）
		self.n_users_test = None      # 用户数（测试集）
		self.n_items_test = None      # 商品数（测试集）

		self.item_pop = {}            # 商品热门度（训练集）, dict, {item: 购买次数, ...}
		self.item_co = {}             # 商品-商品共轭表（训练集）, dict, {item1: {item2: 共同购买过的人数, ...}, ...}
		self.item_sim = {}            # 商品相似度（训练集）, dict, {item1: {item2: 相似度, ...}, ...}

	def load_data(self):
		"""加载数据"""
		eprint("load data...")
		data_path = self.data_path
		columns = ["userId", "movieId", "rating", "timestamp"]

		ratings = pd.read_csv(data_path + "ratings.dat", sep="::", names=columns)
		self.ratings = ratings
		self.n_users = len(ratings["userId"].unique())
		self.n_items = len(ratings["movieId"].unique())

		eprint("load data finished")
		eprint("ratings.head():")
		eprint(ratings.head())
		eprint("ratings's shape {}".format(ratings.shape))
		eprint("n_users {}".format(self.n_users))
		eprint("n_items {}".format(self.n_items))

	def get_train_test(self, test_size=0.3):
		"""获取训练集，测试集（dataframe形式）"""
		eprint("get train and test set...")
		ratings = self.ratings
		seed = self.seed
		train, test = train_test_split(ratings, test_size=test_size, random_state=seed)
		eprint("test size is {}".format(test_size))

		eprint("train shape {}".format(train.shape))
		eprint("train users {}".format(len(train["userId"].unique())))
		eprint("train items {}".format(len(train["movieId"].unique())))

		eprint("test shape {}".format(test.shape))
		eprint("test users {}".format(len(test["userId"].unique())))
		eprint("test items {}".format(len(test["movieId"].unique())))

		self.train = train
		self.test = test
		self.n_users_train = len(train["userId"].unique())
		self.n_items_train = len(train["movieId"].unique())
		self.n_users_test = len(test["userId"].unique())
		self.n_items_test = len(test["movieId"].unique())

	def get_user_item_tbl(self):
		"""计算用户-商品倒排表（训练集，测试集）"""
		eprint("get user-item inverted table...")
		train = self.train
		test = self.test
		train_dict = self.train_dict
		test_dict = self.test_dict

		for i in np.arange(train.shape[0]):
			user = train["userId"].iloc[i]
			item = train["movieId"].iloc[i]
			rating = train["rating"].iloc[i]
			train_dict.setdefault(user, {})
			train_dict[user][item] = int(rating)

		for i in np.arange(test.shape[0]):
			user = test["userId"].iloc[i]
			item = test["movieId"].iloc[i]
			rating = test["rating"].iloc[i]
			test_dict.setdefault(user, {})
			test_dict[user][item] = int(rating)

		eprint("get user-item inverted table finished")
		self.train_dict = train_dict  # dict, {user: {item: rating, ...}, ...}
		self.test_dict = test_dict    # dict, {user: {item: rating, ...}, ...}

	def get_item_pop(self):
		"""计算商品的热门程度，即购买次数（训练集）"""
		eprint("get item pop...")
		train_dict = self.train_dict
		item_pop = self.item_pop
		for user, items in train_dict.items():
			for item in items:
				if item not in item_pop:
					item_pop[item] = 0
				item_pop[item] += 1
		eprint("get item pop finished")
		self.item_pop = item_pop  # {item: 购买次数, ...}

	def get_item_co(self):
		"""计算商品-商品的共轭表"""
		eprint("get item co...")
		item_co = self.item_co
		train_dict = self.train_dict
		for user, items in train_dict.items():
			for i in items:
				item_co.setdefault(i, defaultdict(int))
				for j in items:
					if j == i:
						continue
					item_co[i][j] += 1
		eprint("get item co finished")
		self.item_co = item_co   # {item1: {item2: 共同购买人数, ...}, ...}

	def get_item_sim(self):
		"""计算商品-商品相似度表"""
		eprint("get item sim...")
		item_co = self.item_co
		item_pop = self.item_pop

		cnt = 0
		print_step = 2000000
		for item1, related in item_co.items():
			for item2, item1_item2_cnt in related.items():
				item_co[item1][item2] = item1_item2_cnt / np.sqrt(item_pop[item1] * item_pop[item2])
				cnt += 1
				if cnt % print_step == 0:
					eprint("get item sim cnt {}".format(cnt))
		eprint("get item sim finished")
		self.item_sim = item_co   # {item1: {item2: 相似度, ...}, ...}

	def recommend(self, user, k=20, n=10):
		"""服务推荐，利用训练集计算得到的item_sim，对训练集用户推荐商品"""
		item_sim = self.item_sim
		train_dict = self.train_dict
		items = train_dict[user]
		rcmd = dict()

		for item, rating in items.items():
			for related_item, related_sim in \
					sorted(item_sim[item].items(), key=itemgetter(1), reverse=True)[:k]:
				if related_item in items:
					continue
				rcmd.setdefault(related_item, 0)
				rcmd[related_item] += related_sim * rating

		return sorted(rcmd.items(), key=itemgetter(1), reverse=True)[:n]

	def evaluate(self, k=20, n=10):
		"""推荐效果评价，对训练集所有用户进行商品推荐，并利用测试集作为真实标签进行模型评估"""
		eprint("recommend and evaluate...")
		train_dict = self.train_dict
		test_dict = self.test_dict
		item_pop = self.item_pop
		n_items_train = self.n_items_train
		hit = 0                    # 推荐商品的命中次数
		rcmd_cnt = 0               # 推荐给用户的商品总数
		real_cnt = 0               # 用户购买过的商品总数
		all_rcmd_items = set()     # 推荐给用户的商品总集合
		pop = 0                    # 流行度

		# 对训练集中的用户推荐商品
		for i, user in enumerate(train_dict):
			real_items = test_dict.get(user, {})
			rcmd_items = self.recommend(user)
			for item, _ in rcmd_items:
				if item in real_items:
					hit += 1
				all_rcmd_items.add(item)
				pop += np.log(1 + item_pop[item])
			rcmd_cnt += n
			real_cnt += len(real_items)
			if i % 100 == 0:
				eprint("recommended for {}th users...".format(i))

		precision = hit / (rcmd_cnt * 1.0)
		recall = hit / (real_cnt * 1.0)
		f1_score = 2 * precision * recall / (precision + recall)
		coverage = len(all_rcmd_items) / (n_items_train * 1.0)
		popularity = pop / (rcmd_cnt * 1.0)

		eprint("evaluated results(with k = {}, n = {}):".format(k, n))
		eprint("precision = {:.2%}".format(precision))
		eprint("recall = {:.2%}".format(recall))
		eprint("f1_score = {:.2%}".format(f1_score))
		eprint("coverage = {:.2%}".format(coverage))
		eprint("popularity = {:.3}".format(popularity))


def eprint(*args, **kwargs):
	"""eprint"""
	print(*args, file=sys.stderr, **kwargs)


if __name__ == "__main__":
	itemcf = ItemCf()
	itemcf.load_data()
	itemcf.get_train_test()
	itemcf.get_user_item_tbl()
	itemcf.get_item_pop()
	itemcf.get_item_co()
	itemcf.get_item_sim()
	rcmd = itemcf.recommend(user=1)
	eprint(rcmd)
	itemcf.evaluate()













