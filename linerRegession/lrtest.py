# coding=utf-8
from ml.linerRegession import locallyWeightedLinerRegession
from ml.linerRegession import standRegession
import numpy as np
def rssError(yArr, yHatArr):
    '''
        Desc:
            计算分析预测误差的大小
        Args:
            yArr：真实的目标变量
            yHatArr：预测得到的估计值
        Returns:
            计算真实值和估计值得到的值的平方和作为最后的返回值
    '''
    return ((yArr - yHatArr) ** 2).sum()
def abaloneTest():
    '''
    Desc:
        预测鲍鱼的年龄
    Args:
        None
    Returns:
        None
    '''
    # 加载数据
    abX, abY = standRegession.load_data_file("abalone.txt")
    print(len(abX))
    # # 使用不同的核进行预测
    # oldyHat01 = locallyWeightedLinerRegession.lwlrTest(abX[0:999], abX[0:999], abY[0:999], 0.1)
    # oldyHat1 = locallyWeightedLinerRegession.lwlrTest(abX[0:999], abX[0:999], abY[0:999], 1)
    # oldyHat10 = locallyWeightedLinerRegession.lwlrTest(abX[0:999], abX[0:999], abY[0:999], 10)
    # # 打印出不同的核预测值与训练数据集上的真实值之间的误差大小
    # print("old yHat01 error Size is :", rssError(abY[0:999], oldyHat01.T))
    # print("old yHat1 error Size is :", rssError(abY[0:999], oldyHat1.T))
    # print("old yHat10 error Size is :", rssError(abY[0:999], oldyHat10.T))

    # 打印出 不同的核预测值 与 新数据集（测试数据集）上的真实值之间的误差大小
    # newyHat01 = locallyWeightedLinerRegession.lwlrTest(abX[1000:1999], abX[0:999], abY[0:999], 0.1)
    # print("new yHat01 error Size is :", rssError(abY[1000:1999], newyHat01.T))
    # newyHat1 = locallyWeightedLinerRegession.lwlrTest(abX[1000:1999], abX[0:999], abY[0:999], 1)
    # print("new yHat1 error Size is :", rssError(abY[1000:1999], newyHat1.T))
    # newyHat10 = locallyWeightedLinerRegession.lwlrTest(abX[1000:1999], abX[0:999], abY[0:999], 10)
    # print("new yHat10 error Size is :", rssError(abY[1000:1999], newyHat10.T))

    # 使用简单的 线性回归 进行预测，与上面的计算进行比较
    standWs = standRegession.stand_liner_regession(abX[0:999], abY[0:999])
    standyHat = np.mat(abX[1000:1500]) * standWs
    print("standRegress error Size is:", rssError(abY[1000:1500], standyHat.T.A))

if __name__ == '__main__':
    # regression1()
    # regression2()
     abaloneTest()