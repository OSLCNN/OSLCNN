import random

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']



def curve_flops_cpu_43():
    # df = pd.read_excel('datasets/conv_cpu_flops_130_4300.xls')
    df = pd.read_excel('datasets/linear_cpu_flops_3_260.xls')

    df.sample(frac=1)   # 打乱顺序
    x = np.array(df['flops'].tolist())
    y = np.array(df['cpu'].tolist())
    z = np.array(df['latency'].tolist())

    x_train = []
    y_train = []
    z_train = []
    x_test = []
    y_test = []
    z_test = []
    num = range(len(df))
    randomNum = random.sample(num, int(len(df) * 0.8))
    for i in num:
        if i in randomNum:
            x_train.append(x[i])
            y_train.append(y[i])
            z_train.append(z[i])
        else:
            x_test.append(x[i])
            y_test.append(y[i])
            z_test.append(z[i])

    # 第一步，拟合
    xy = np.vstack((x, y))
    xy_train = np.vstack((x_train, y_train))
    xy_test = np.vstack((x_test, y_test))

    def Pfun(t, a1, b1, a2, b2, c2, d2, e2, f2):
        return (a1 * t[0] + b1) * \
               (a2 * np.power(t[1], 5) + b2 * np.power(t[1], 4) + c2 * np.power(t[1], 3) +
                d2 * np.power(t[1], 2) + e2 * t[1] + f2)
    popt, pcov = curve_fit(Pfun, xy_train, z_train)  # 拟合参数
    print("拟合值分别为：", popt)
    # 画图
    ax = plt.axes(projection='3d')  # 创建一个三维坐标轴曲对象
    ax.scatter(x, y, z, alpha=0.3)
    plt.rc('font', size=16)

    z_train_predict = Pfun(xy_train, *popt)
    z_test_predict = Pfun(xy_test, *popt)
    print('MAE train: %.3f, test: %.3f' %
         (mean_absolute_error(z_train, z_train_predict),
         mean_absolute_error(z_test, z_test_predict)))

    print('MAPE train: %.3f, test: %.3f' %
         (mean_absolute_percentage_error(z_train, z_train_predict),
         mean_absolute_percentage_error(z_test, z_test_predict)))

    print('R^2 train: %.3f, test: %.3f' %
          (r2_score(z_train, z_train_predict),
           r2_score(z_test, z_test_predict)))

    z_predict = Pfun(xy, *popt)
    ax.scatter(x, y, z_predict, alpha=0.3)

    ax.set_xlabel('flops(M)')
    ax.set_ylabel('cpu(%)')
    ax.set_zlabel('latency(ms)')

    plt.show()



if __name__ == '__main__':

    curve_flops_cpu_43()
