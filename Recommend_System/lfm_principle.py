# !/usr/bin/env python
# coding=utf-8

"""
	@author:	xin.jin
	@file:		lfm_principle.py
	@time:		2019.07.08
	@notes:		
	
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def lfm(A, k):
	"""
	:param A: 表示需要分解的评价矩阵
	:param k: 分解的属性（隐变量）个数
	:return:
	"""
	assert type(A) == np.ndarray

	alpha = 0.01                 # 学习率
	lambda_ = 0.01               # 正则化参数
	epochs = 10000               # 随机梯度下降的次数
	print_step = 200             # 记录代价的间隔
	cost_list = []               # 记录代价的list
	epoch_list = []

	m, n = A.shape               # 用户数m，商品数n
	P = np.random.rand(m, k)     # "个人口味"矩阵
	Q = np.random.randn(k, n)    # "属性分值"矩阵

	for epoch in range(epochs):  # 进行epochs轮随机梯度下降
		for i in range(m):  # 遍历m个用户
			for j in range(n):  # 遍历n个商品
				if math.fabs(A[i][j]) > 1e-4:
					err = A[i][j] - np.dot(P[i, :], Q[:, j])  # 真实评分和预测平分的差值
					# 遍历商品的k个属性
					for r in range(k):
						grad_P = err * Q[r][j] - lambda_ * P[i][r]  # J对P[i][r]的负梯度
						grad_Q = err * P[i][r] - lambda_ * Q[r][j]  # J对Q[r][j]的负梯度
						P[i][r] += alpha * grad_P                   # P梯度下降
						Q[r][j] += alpha * grad_Q                   # Q梯度下降
		# 计算代价
		cost = calc_cost(A, P, Q, k, lambda_)
		if epoch % print_step == 0:
			print("epoch {}, cost = {}...".format(epoch, cost))
			cost_list.append(cost)
			epoch_list.append(epoch)
	# 绘制learning curve
	plot_cost(epoch_list, cost_list)
	return P, Q, cost_list, epoch_list


def calc_cost(A, P, Q, k, lambda_reg):
	"""计算代价值"""
	cost_err = np.sum(np.power(A - np.dot(P, Q), 2))
	cost_reg_P = lambda_reg * np.sum(np.power(P, 2))
	cost_reg_Q = lambda_reg * np.sum(np.power(Q, 2))
	cost = cost_err + cost_reg_P + cost_reg_Q
	return cost


def plot_cost(epoch_list, cost_list):
	"""绘制学习曲线"""
	plt.plot(epoch_list, cost_list)
	plt.xlabel("epoch")
	plt.ylabel("cost")
	plt.title("learning curve")
	plt.show()


if __name__ == "__main__":
	A = np.array([[5, 0, 1, 0, 5],
	              [4, 2, 0, 0, 5],
	              [0, 5, 1, 2, 4],
	              [5, 0, 5, 3, 2]])
	P, Q, cost_list, epoch_list = lfm(A, 3)

	# 查看结果
	print(P)             # "个人口味"
	print(Q)             # "属性分值"
	print(np.dot(P, Q))  # 预测评分
	print(cost_list)
	print(epoch_list)





