# coding=utf-8

"""
创建随机数据，理解sklearn的决策树用法
决策树回归的用法
"""
# 引入必要的模型和库
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pylab as plt

# 创建一个随机数据集
rng = np.random.RandomState(1)
# rand() 是给定形状的随机值，rng.rand(80, 1)即矩阵的形状是 80行，1列
# sort()
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
print(y.shape)
print(y[::5])
y[::5] += 3 * (0.5 - rng.rand(16))
# 拟合回归模型

regr_1 = DecisionTreeRegressor(max_depth=2)
# 保持 max_depth=5 不变，增加 min_samples_leaf=6 的参数，效果进一步提升了
regr_2 = DecisionTreeRegressor(max_depth=5)
# regr_2 = DecisionTreeRegressor(min_samples_leaf=6)
# regr_3 = DecisionTreeRegressor(max_depth=4)
regr_1.fit(X, y)
# regr_2.fit(X, y)
# regr_3.fit(X, y)

# 预测
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
# y_2 = regr_2.predict(X_test)
# y_3 = regr_3.predict(X_test)
# 绘制结果

plt.figure()
plt.scatter(X, y, c="darkorange", label="data")
plt.plot(X_test, y_1, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
