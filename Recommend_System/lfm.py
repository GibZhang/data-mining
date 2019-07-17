# !/usr/bin/env python
# coding=utf-8

"""
	@author:	xin.jin
	@file:		lfm.py
	@time:		2019.07.04
	@notes:		
	
"""

import numpy as np
import pandas as pd
import os
import sys
import warnings
import collections
import random

from sklearn.model_selection import train_test_split
from operator import itemgetter

random_seed = 0
warnings.filterwarnings("ignore")


class LFM(object):
	"""Latent Factor Model隐语义模型"""

	def __init__(self):
		self.seed = 0
		self.data_path = r"E:\MyProgramming\python_workspace\55. recommendation\my rcmd\data_movieLens" + os.sep
		self.data_choose = None

		# 总数据
		self.ratings = None              # 用户商品评分表（总）, df
		self.n_users = None              # 用户数（总）
		self.n_items = None              # 商品数（总）

		# 训练集，测试集数据
		self.train = None                # 用户商品评分表（训练集）, df
		self.test = None                 # 用户商品评分表（测试集）, df

		self.train_user2item = dict()    # 用户-商品倒排表（训练集）, dict, {user: {item: rating, ...}, ...}
		self.test_user2item = dict()     # 用户-商品倒排表（测试集）, dict, {user: {item: rating, ...}, ...}

		self.train_item2user = dict()    # 商品-用户倒排表（训练集）, dict, {item1: (user1, ...), ...}
		self.test_item2user = dict()     # 商品-用户倒排表（测试集）, dict, {item1: (user1, ...), ...}

		self.n_users_train = None        # 用户数（训练集）
		self.n_items_train = None        # 商品数（训练集）
		self.n_users_test = None         # 用户数（测试集）
		self.n_items_test = None         # 商品数（测试集）

		self.item_pop = dict()           # 商品热门度（训练集）, dict, {item: 购买次数, ...}
		self.item_pool = list()          # 商品购买记录表（训练集）, list, [item1, item1, item2, item3, ...]
		self.user_co = {}                # 商品-商品共轭表（训练集）, dict, {user1: {user2: 共同购买过的商品数, ...}, ...}
		self.user_sim = {}               # 商品相似度（训练集）, dict, {user1: {user2: 相似度, ...}, ...}

		# lfm模型相关
		self.k = 10                      # 隐特征个数
		self.n_rec_movie = 10            # 推荐商品个数
		self.epochs = 10                 # 训练集分组数
		self.alpha = 0.1                 # 学习率
		self.lambda_reg = 0.01           # 正则化参数
		self.ratio = 10                  # 负样本/正样本比例

		self.P = dict()                  # 用户对隐特征的兴趣，P[user]为k维list
		self.Q = dict()                  # 隐特征和商品的关系，Q[item]为k维list

	def load_data(self, file="ml-100k"):
		"""加载数据"""
		eprint("load data...")
		self.data_choose = file
		data_path = self.data_path + file + os.sep
		columns = ["userId", "movieId", "rating", "timestamp"]
		try:
			if file == "ml-latest-small":
				ratings = pd.read_csv(data_path + "ratings.csv")
			if file == "ml-100k":
				ratings = pd.read_csv(data_path + "u.data.csv", names=columns)
			elif file == "ml-1m" or file == "ml-10m":
				ratings = pd.read_csv(data_path + "ratings.dat", sep="::", names=columns)
		except OSError as e:
			eprint("load data failed, no such data!")
			raise e

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

	def get_train_test_dict(self):
		"""获取train, test的user-item倒排表和item-user倒排表"""
		eprint("get train test dict...")
		train = self.train
		test = self.test

		train_user2item = self.train_user2item
		test_user2item = self.test_user2item

		train_item2user = self.train_item2user
		test_item2user = self.test_item2user

		item_pool = self.item_pool

		eprint("get train user2item...")
		for i in np.arange(train.shape[0]):
			user = train["userId"].iloc[i]
			item = train["movieId"].iloc[i]
			rating = train["rating"].iloc[i]
			train_user2item.setdefault(user, {})
			train_user2item[user][item] = int(rating)
			item_pool.append(item)  # 商品购买记录表，有重复（训练集）, list, [item1, item1, item2, item3, ...]

		eprint("get test user2item...")
		for i in np.arange(test.shape[0]):
			user = test["userId"].iloc[i]
			item = test["movieId"].iloc[i]
			rating = test["rating"].iloc[i]
			test_user2item.setdefault(user, {})
			test_user2item[user][item] = int(rating)

		eprint("get train item2user...")
		for user, items in train_user2item.items():
			for item in items:
				if item not in train_item2user:
					train_item2user[item] = set()
				train_item2user[item].add(user)

		eprint("get test item2user...")
		for user, items in test_user2item.items():
			for item in items:
				if item not in test_item2user:
					test_item2user[item] = set()
				test_item2user[item].add(user)

		eprint("get train test dict finished")
		self.train_user2item = train_user2item
		self.test_user2item = test_user2item
		self.train_item2user = train_item2user
		self.test_item2user = test_item2user
		self.item_pool = item_pool

	def get_item_pop(self):
		"""计算商品的热门程度，即购买次数（训练集）"""
		eprint("get item pop...")
		train_user2item = self.train_user2item
		item_pop = self.item_pop
		for user, items in train_user2item.items():
			for item in items:
				if item not in item_pop:
					item_pop[item] = 0
				item_pop[item] += 1
		eprint("get item pop finished")
		self.item_pop = item_pop  # {item: 购买次数, ...}

	def init_latent(self):
		"""初始化隐向量"""
		eprint("init latent...")
		P = self.P
		Q = self.Q
		k = self.k
		train_user2item = self.train_user2item
		train_item2user = self.train_item2user
		for user in train_user2item:
			P[user] = [np.random.random() / np.sqrt(k) for _ in range(k)]
		for item in train_item2user:
			Q[item] = [np.random.random() / np.sqrt(k) for _ in range(k)]
		eprint("init latent finished")
		self.P = P
		self.Q = Q

	def generate_negative_samples(self, items: dict):
		"""
		生成负样本
		:param items: train_user2item[user]，是某一个用户购买过的商品dict
		:return:
		"""
		item_pool = self.item_pool  # 训练集中商品的出现列表（有重复），商品i出现次数越多，表示商品i越流行，越容易被采样到
		ratio = self.ratio
		resampled_items = dict()    # {item: 1 or 0, ...}，其中1表示正样本，0表示负样本

		for item, rating in items.items():
			resampled_items[item] = 1     # 某用户user购买过的商品标记为1
		for i in range(len(items) * 11):  # 这里11是一个比较大的数，防止每次重采样的结果都是user购买过的商品
			item = item_pool[random.randint(0, len(item_pool) - 1)]  # 这里item是item_poor中比较热门的商品
			if item in resampled_items:
				continue
			resampled_items[item] = 0  # 用户user没有购买过的，且比较热门的标记为0
			if (len(resampled_items)-len(items)) / len(items) >= ratio:  # 负样本/正样本的比例ratio
				break
		return resampled_items

	def predict(self, user, item):
		"""
		根据P, Q预测user对item的打分
		:param user: 某用户
		:param item: 某商品
		:return: user对item的预测打分
		"""
		P = self.P
		Q = self.Q
		k = self.k
		rating_pred = 0
		for i in range(k):
			p = P[user][i]  # P[user]是k维向量
			q = Q[item][i]  # Q[item]是k维向量
			rating_pred += p * q
		return rating_pred

	def train_model(self):
		"""
		对训练集进行模型训练
		遗留问题：考虑一下如何计算cost？
		:return:
		"""
		eprint("training...")
		train_user2item = self.train_user2item
		train_item2user = self.train_item2user
		epochs = self.epochs
		k = self.k
		lambda_reg = self.lambda_reg

		# 共进行epochs次循环
		for epoch in range(epochs):

			# 对训练集中所有的user进行遍历
			for user in train_user2item:
				items = train_user2item[user]  # user购买过的商品, {item: rating, ...}
				resampled_items = self.generate_negative_samples(items)  # items重采样之后的商品, {item: 0 or 1, ...}

				# 对用户user重采样集中的所有pos/neg item进行遍历
				for item, rui in resampled_items.items():
					rui_pred = self.predict(user, item)  # user对某商品item的预测评分
					eui = rui - rui_pred  # 差值 = 1 or 0 - 预测评分

					# 梯度下降，更新: P[user][i], Q[item][i]
					for i in range(k):
						self.P[user][i] += self.alpha * (eui * self.Q[item][i] - lambda_reg * self.P[user][i])
						self.Q[item][i] += self.alpha * (eui * self.P[user][i] - lambda_reg * self.Q[item][i])

			self.alpha *= 0.9  # 每训练一轮epoch，学习率降低
			eprint("training epoch {} ...".format(epoch))

		eprint("training finished")

	def recommend(self, user):
		"""推荐商品"""
		train_user2item = self.train_user2item
		item_pop = self.item_pop
		P = self.P
		Q = self.Q
		n = self.n_rec_movie

		rcmd = collections.defaultdict(float)
		bought_items = train_user2item[user]
		for item in item_pop:
			if item in bought_items.keys():
				continue
			for k, Qik in enumerate(Q[item]):
				rcmd[item] += P[user][k] * Qik
		return [item for item, _ in sorted(rcmd.items(), key=itemgetter(1), reverse=True)][:n]

	def evaluate(self):
		"""推荐效果评价，对训练集所有用户进行商品推荐，并利用测试集作为真实标签进行模型评估"""
		eprint("recommend and evaluate...")
		data_choose = self.data_choose
		train_user2item = self.train_user2item
		test_user2item = self.test_user2item
		item_pop = self.item_pop
		n_items_train = self.n_items_train
		k = self.k
		n = self.n_rec_movie
		lambda_reg = self.lambda_reg
		ratio = self.ratio

		hit = 0                  # 推荐商品的命中次数
		rcmd_cnt = 0             # 推荐给用户的商品总数
		real_cnt = 0             # 用户购买过的商品总数
		all_rcmd_items = set()   # 推荐给用户的商品总集合
		pop = 0                  # 流行度

		# 对训练集中的用户推荐商品
		for i, user in enumerate(train_user2item):
			real_items = test_user2item.get(user, {})
			rcmd_items = self.recommend(user)
			for item in rcmd_items:
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

		eprint("evaluated results"
		       "(with data = {}, "
		       "k = {}, "
		       "n = {}, "
		       "lambda_reg = {}, "
		       "ratio = {}):".format(data_choose, k, n, lambda_reg, ratio))

		eprint("precision = {:.2%}".format(precision))
		eprint("recall = {:.2%}".format(recall))
		eprint("f1_score = {:.2%}".format(f1_score))
		eprint("coverage = {:.2%}".format(coverage))
		eprint("popularity = {:.3}".format(popularity))

		eprint("evaluate finished")


def eprint(*args, **kwargs):
	"""eprint"""
	print(*args, file=sys.stderr, **kwargs)


if __name__ == "__main__":
	lfm = LFM()

	# 选择数据, ml-latest-small, ml-100k, ml-1m
	lfm.load_data(file="ml-100k")

	lfm.get_train_test()
	lfm.get_train_test_dict()
	lfm.get_item_pop()
	lfm.init_latent()
	lfm.train_model()
	lfm.evaluate()










