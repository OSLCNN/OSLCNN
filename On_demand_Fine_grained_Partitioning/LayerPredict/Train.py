import random
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
import joblib


def knn():
    # 训练KNN
    clf = KNeighborsRegressor(n_neighbors=1, p=2, metric='minkowski')
    return clf


def svm_function():
    clf = svm.SVR()
    return clf


def dt():
    clf = DecisionTreeRegressor()
    return clf


def rf():
    clf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=0)
    return clf


def ada():
    # 注意要修改数据集，转为整型
    # z = np.array(df['latency'].astype('int').tolist())
    clf = AdaBoostClassifier(n_estimators=180, random_state=0)
    return clf


def krr():
    clf = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                      param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                                  "gamma": np.logspace(-2, 2, 5)})
    return clf

# 画图
def plot_cpu_flops():
    file_name = 'datasets/linear_cpu_flops_3_260.xls'
    df = pd.read_excel(file_name)
    # 定义坐标轴
    ax = plt.axes(projection='3d')
    # 生成三维数据
    xx = np.array(df['flops'].tolist())
    yy = np.array(df['cpu'].tolist())
    Z = np.array(df['latency'].tolist())
    ax.scatter(xx, yy, Z, alpha=0.3 )  # 生成散点.利用c控制颜色序列,s控制大小
    ax.set_xlabel('flops(M)')
    ax.set_ylabel('cpu(%)')
    ax.set_zlabel('latency(ms)')
    plt.show()


def predict():
    # df = pd.read_excel('datasets/conv_cpu_flops_130_4300.xls')
    df = pd.read_excel('datasets/linear_cpu_flops_3_260.xls')
    df.sample(frac=1)  # 打乱顺序
    x = np.array(df['flops'].tolist())
    y = np.array(df['cpu'].tolist())
    z = np.array(df['latency'].tolist())
    # z = np.array(df['latency'].astype('int').tolist())

    xy_train = []
    z_train = []
    xy_test = []
    z_test = []
    xy = []
    x_test = []
    y_test = []
    num = range(len(df))
    randomNum = random.sample(num, int(len(df) * 0.8))
    for i in num:
        if i in randomNum:
            xy_train.append([x[i], y[i]])
            z_train.append(z[i])
        else:
            xy_test.append([x[i], y[i]])
            z_test.append(z[i])
            x_test.append(x[i])
            y_test.append(y[i])
        xy.append([x[i], y[i]])

    ax = plt.axes(projection='3d')  # 创建一个三维坐标轴曲对象
    ax.scatter(x, y, z, alpha=0.3)
    plt.rc('font', size=16)

    clf = knn()
    clf.fit(xy_train, z_train)  # 训练
    joblib.dump(clf, "linear_rf_model.m")   # 存储

    z_train_predict = clf.predict(xy_train)     # 预测
    z_test_predict = clf.predict(xy_test)
    print('MAE train: %.3f, test: %.3f' %
          (mean_absolute_error(z_train, z_train_predict),
           mean_absolute_error(z_test, z_test_predict)))

    print('MAPE train: %.3f, test: %.3f' %
          (mean_absolute_percentage_error(z_train, z_train_predict),
           mean_absolute_percentage_error(z_test, z_test_predict)))

    print('R^2 train: %.3f, test: %.3f' %
          (r2_score(z_train, z_train_predict),
           r2_score(z_test, z_test_predict)))

    z_predict = clf.predict(xy)
    ax.scatter(x, y, z_predict, alpha=0.3)

    ax.set_xlabel('flops(M)')
    ax.set_ylabel('cpu(%)')
    ax.set_zlabel('latency(ms)')

    plt.show()
    return clf


def Pfun(t, a1, b1, a2, b2, c2, d2, e2, f2):
    return (a1 * t[0] + b1) * \
           (a2 * np.power(t[1], 5) + b2 * np.power(t[1], 4) + c2 * np.power(t[1], 3) +
            d2 * np.power(t[1], 2) + e2 * t[1] + f2)


def curve():    # 多项式拟合
    #df = pd.read_excel('datasets/linear_cpu_flops_3_260.xls')
    df = pd.read_excel('datasets/conv_cpu_flops_130_4300.xls')

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

conv_popt = [1.04706796e-07,  6.52876925e-06,  4.93731383e-03, -9.59855386e-01,
             6.57436585e+01, -1.83081160e+03,  1.98693668e+04,  6.79433621e+04]
fc_popt = [2.28457750e-09, -4.25494006e-09,  8.06625307e-01, -1.40473702e+02,
           8.88610637e+03, -2.08845416e+05,  2.37566153e+06,  7.15240467e+07]

def predict():
    flops = 0.629
    cpu = 100
    xy = np.vstack((np.array([flops]), np.array([cpu])))
    z_predict = Pfun(xy, *conv_popt)
    print(z_predict)
    # print(z_predict * 17)

if __name__ == '__main__':
    # 画图
    # train_other_model()
    predict()
    curve()