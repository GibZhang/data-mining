# coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns


def load_data(filename):
    """
    从文件读取鸢尾花数据，文件格式为csv
    :param filename: iris.csv
    """
    df = pd.read_csv('iris.csv')
    return df


def iris_feature_plot(df):
    """
    画出鸢尾花各属性之间的点位图
    计算各属性的基本数据
    :param df:
    """
    fig, axes = plt.subplots(1, 4, figsize=(12, 7))
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    ax4 = axes[3]
    sns.boxplot('Species', 'Sepal length', data=df, hue='Species', ax=ax1)
    sns.boxplot('Species', 'Sepal width', data=df, hue='Species', ax=ax2)
    sns.boxplot('Species', 'Petal length', data=df, hue='Species', ax=ax3)
    sns.boxplot('Species', 'Petal width', data=df, hue='Species', ax=ax4)
    # plt.savefig('/Users/zhangjingbo/Desktop/数据分析案例/机器学习算法/decisionTree/box.png')
    sns.pairplot(data=df, hue='Species', x_vars=["Sepal width", "Sepal length"],
                 y_vars=["Petal width", "Petal length"], palette=sns.color_palette("muted"), height=2.5)
    sns.pairplot(data=df, hue='Species', x_vars=["Sepal width", "Petal width"],
                 y_vars=["Sepal length", "Petal length"], palette=sns.color_palette("muted"), height=2.5)
    sns.set(style="ticks", color_codes=True)
    iris = sns.load_dataset("iris",data_home='')
    g = sns.pairplot(iris,hue='species')
    plt.show()


if __name__ == '__main__':
    df = load_data('iris.csv')
    iris_feature_plot(df)
