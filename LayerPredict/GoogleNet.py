import os
import random
import sys
from datetime import datetime
import time
import pandas as pd
from scipy.optimize import curve_fit
import joblib
import psutil
import numpy as np
import torch
import torch.nn as nn

conv_popt = 0
linear_popt = 0
_timer = getattr(time, 'monotonic', time.time)  # 获取对象time的monotonic属性，如果属性不存在，默认值为time.time
num_cpus = psutil.cpu_count() or 1
pid = 0

# 获取当前进程PID
def get_pid():
    f = os.popen("pidof python GenerateData.py").read().split(' ')[0]
    print(f)
    return int(f)

def timer():
    return _timer()

def getCpuAndTime(pid):
    # 获取当前时间和cpu利用率
    p = psutil.Process(pid)
    pt = p.cpu_times()
    st1, pt1_0, pt1_1 = timer(), pt.user, pt.system  # new
    pt1 = pt1_0 + pt1_1
    return st1, pt1


class AlexNet(nn.Module):
    def __init__(self,num_classes=1000):
        super(AlexNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2, bias=False),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=0)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=0)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.drop1 = nn.Dropout(p=0.5)

        self.linear1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)

        self.drop2 = nn.Dropout(p=0.5)

        self.linear2 = nn.Linear(in_features=4096, out_features=4096)

        self.linear3 = nn.Linear(in_features=4096, out_features=num_classes)

        self.total_real_time = 0
        self.total_predict_time = 0

    def forward(self, x):
        global pid
        pid = get_pid()
        self.total_real_time = 0
        self.total_real_time = 0
        x = self.conv_func(x, "conv1")
        x = self.pool1(x)
        x = self.conv_func(x, "conv2")
        x = self.pool2(x)
        x = self.conv_func(x, "conv3")
        x = self.conv_func(x, "conv4")
        x = self.conv_func(x, "conv5")
        x = self.pool3(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.drop1(x)

        x = self.linear_func(x, "linear1")
        x = self.drop2(x)
        x = self.linear_func(x, "linear2")
        x = self.linear_func(x, "linear3")

        print("总时间：" + str(self.total_real_time) + ", 预测：" + str(self.total_predict_time) + "差值：" +
              str(abs(self.total_real_time-self.total_predict_time[0])))
        return x

    def conv_func(self, x, name):
        in_channel = x.shape[1]
        count = 30       # 当执行时间太短无法测量cpu时，会增加count，共执行count次
        pre_x = x
        if name == "conv1":
            start_time, start_cpu = getCpuAndTime(pid)
            for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                x = self.conv1(pre_x)
            end_time, end_cpu = getCpuAndTime(pid)
            k = 11
        elif name == "conv2":
            start_time, start_cpu = getCpuAndTime(pid)
            for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                x = self.conv2(pre_x)
            end_time, end_cpu = getCpuAndTime(pid)
            k = 5
        elif name == "conv3":
            start_time, start_cpu = getCpuAndTime(pid)
            for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                x = self.conv3(pre_x)
            end_time, end_cpu = getCpuAndTime(pid)
            k = 3
        elif name == "conv4":
            start_time, start_cpu = getCpuAndTime(pid)
            for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                x = self.conv4(pre_x)
            end_time, end_cpu = getCpuAndTime(pid)
            k = 3
        elif name == "conv5":
            start_time, start_cpu = getCpuAndTime(pid)
            for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                x = self.conv5(pre_x)
            end_time, end_cpu = getCpuAndTime(pid)
            k = 3
        real_time = (end_time - start_time) / count
        cpu = round((end_cpu - start_cpu) / (end_time - start_time) * num_cpus * 100, 2)
        flops = get_conv_flops(in_channel, x.shape[1], x.shape[2], k)
        xy = np.vstack((np.array([flops]), np.array([cpu])))
        predict_time = Pfun(xy, *conv_popt) / 1000
        self.total_real_time += real_time
        self.total_predict_time += predict_time
        print(name + " ---  flops: " + str(flops) + "cpu: " + str(cpu) + "real: " + str(real_time) +
              "predict: " + str(predict_time))
        return x

    def linear_func(self, x, name):
        in_channel = x.shape[1]
        count = 30
        pre_x = x
        if name == "linear1":
            start_time, start_cpu = getCpuAndTime(pid)
            for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                x = self.linear1(pre_x)
            end_time, end_cpu = getCpuAndTime(pid)
        elif name == "linear2":
            start_time, start_cpu = getCpuAndTime(pid)
            for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                x = self.linear2(pre_x)
            end_time, end_cpu = getCpuAndTime(pid)
        elif name == "linear3":
            start_time, start_cpu = getCpuAndTime(pid)
            for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                x = self.linear3(pre_x)
            end_time, end_cpu = getCpuAndTime(pid)
        real_time = (end_time - start_time) / count
        cpu = round((end_cpu - start_cpu) / (end_time - start_time) * num_cpus * 100, 2)
        flops = get_linear_flops(in_channel, x.shape[1])
        xy = np.vstack((np.array([flops]), np.array([cpu])))
        predict_time = Pfun(xy, *linear_popt) / 1000
        print(name + " ---  flops: " + str(flops) + "cpu: " + str(cpu) + "real: " + str(real_time) +
              "predict: " + str(predict_time))
        return x


def get_conv_flops(in_channel, out_channel, output_size, kernel_size):
    flops = 2 * output_size * output_size * (in_channel * kernel_size * kernel_size + 1) * out_channel / (10 ** 6)
    return flops


def get_linear_flops(input_size, output_size):
    flops = (2 * input_size - 1) * output_size / (10 ** 6)
    return flops


def Pfun(t, a1, b1, a2, b2, c2, d2, e2, f2):
        return (a1 * t[0] + b1) * \
               (a2 * np.power(t[1], 5) + b2 * np.power(t[1], 4) + c2 * np.power(t[1], 3) +
                d2 * np.power(t[1], 2) + e2 * t[1] + f2)


def conv_curve():
    df = pd.read_excel('datasets/conv_cpu_flops_minor.xls')
    df.sample(frac=1)  # 打乱顺序
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
    xy = np.vstack((x, y))
    popt, pcov = curve_fit(Pfun, xy, z)  # 拟合参数
    global conv_popt
    conv_popt = popt


def linear_curve():
    df = pd.read_excel('datasets/linear_cpu_flops.xls')
    df.sample(frac=1)  # 打乱顺序
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
    xy = np.vstack((x, y))
    popt, pcov = curve_fit(Pfun, xy, z)  # 拟合参数
    global linear_popt
    linear_popt = popt


if __name__ =='__main__':

    # 无用代码，留出时间来控制CPU利用率
    t1 = datetime.now()
    print("Start : " + str(t1))
    d = 0
    num = 0
    while d < 15:
        num += 1
        t2 = datetime.now()
        d = (t2 - t1).seconds
    print("End : " + str(t2))

    conv_curve()
    linear_curve()

    model = AlexNet()
    input = torch.randn(1,3,224,224)
    out = model(input)

