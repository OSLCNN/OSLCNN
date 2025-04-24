import os
import random
from datetime import datetime
import time
import pandas as pd
from scipy.optimize import curve_fit
import psutil
import numpy as np
import torch
import torch.nn as nn

conv_popt = 0
linear_popt = 0
_timer = getattr(time, 'monotonic', time.time)  # 获取对象time的monotonic属性，如果属性不存在，默认值为time.time
num_cpus = psutil.cpu_count() or 1
# pid = 0

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

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,blocks, num_classes=1000, expansion = 4):
        super(ResNet,self).__init__()
        self.expansion = expansion

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64, kernel_size=7,stride=2, padding=3, bias=False),
            nn.ReLU(inplace=True))

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv1_1_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        # 1_1_4
        self.conv1_1_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64 * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # 1_3_1
        self.conv1_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=64*self.expansion, out_channels=64, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )
        # 1_3_2
        self.conv1_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        # 1_3_3
        self.conv1_2_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64 * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.conv2_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv2_1_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        # 2_1_4
        self.conv2_1_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128 * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # 2_3_1, 2_4_1
        self.conv2_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=128 * self.expansion, out_channels=128, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )
        # 2_3_2, 2_4_2
        self.conv2_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        # 2_3_3, 2_4_3
        self.conv2_2_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128 * self.expansion, kernel_size=1, stride=1,
                      bias=False),
            nn.ReLU(inplace=True),
        )

        self.conv3_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv3_1_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        # 3_1_4
        self.conv3_1_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256 * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # 3_3_1, 3_4_1, 3_5_1, 3_6_1
        self.conv3_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=256 * self.expansion, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )
        # 3_3_2, 3_4_2, 3_5_2, 3_6_2
        self.conv3_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        # 3_3_3, 3_4_3, 3_5_2, 3_6_2
        self.conv3_2_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256 * self.expansion, kernel_size=1, stride=1,
                      bias=False),
            nn.ReLU(inplace=True),
        )

        self.conv4_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv4_1_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        # 4_1_4
        self.conv4_1_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512 * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # 4_3_1, 4_4_1
        self.conv4_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=512 * self.expansion, out_channels=512, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )
        # 4_3_2, 4_4_2
        self.conv4_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        # 4_3_3, 4_4_3
        self.conv4_2_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512 * self.expansion, kernel_size=1, stride=1,
                      bias=False),
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.linear1 = nn.Linear(2048, num_classes)

        self.total_real_time = 0
        self.total_predict_time = 0


    def forward(self, x):

        x = self.conv_func(x, "conv0")
        x = self.pool1(x)
        print(x.shape)
        x = self.conv_func(x, "conv1_1_1")
        x = self.conv_func(x, "conv1_1_2")
        x = self.conv_func(x, "conv1_1_3")
        print(x.shape)
        # x = self.conv_func(x, "conv1_1_4")
        x = self.conv_func(x, "conv1_2_1")
        x = self.conv_func(x, "conv1_2_2")
        x = self.conv_func(x, "conv1_2_3")
        x = self.conv_func(x, "conv1_3_1")
        x = self.conv_func(x, "conv1_3_2")
        x = self.conv_func(x, "conv1_3_3")
        print(x.shape)

        x = self.conv_func(x, "conv2_1_1")
        x = self.conv_func(x, "conv2_1_2")
        x = self.conv_func(x, "conv2_1_3")
        print(x.shape)
        # x = self.conv_func(x, "conv2_1_4")
        x = self.conv_func(x, "conv2_2_1")
        x = self.conv_func(x, "conv2_2_2")
        x = self.conv_func(x, "conv2_2_3")
        print(x.shape)
        x = self.conv_func(x, "conv2_3_1")
        x = self.conv_func(x, "conv2_3_2")
        x = self.conv_func(x, "conv2_3_3")
        print(x.shape)
        x = self.conv_func(x, "conv2_4_1")
        x = self.conv_func(x, "conv2_4_2")
        x = self.conv_func(x, "conv2_4_3")
        print(x.shape)

        x = self.conv_func(x, "conv3_1_1")
        x = self.conv_func(x, "conv3_1_2")
        x = self.conv_func(x, "conv3_1_3")
        # x = self.conv_func(x, "conv3_1_4")
        x = self.conv_func(x, "conv3_2_1")
        x = self.conv_func(x, "conv3_2_2")
        x = self.conv_func(x, "conv3_2_3")
        x = self.conv_func(x, "conv3_3_1")
        x = self.conv_func(x, "conv3_3_2")
        x = self.conv_func(x, "conv3_3_3")
        x = self.conv_func(x, "conv3_4_1")
        x = self.conv_func(x, "conv3_4_2")
        x = self.conv_func(x, "conv3_4_3")
        x = self.conv_func(x, "conv3_5_1")
        x = self.conv_func(x, "conv3_5_2")
        x = self.conv_func(x, "conv3_5_3")
        x = self.conv_func(x, "conv3_6_1")
        x = self.conv_func(x, "conv3_6_2")
        x = self.conv_func(x, "conv3_6_3")

        x = self.conv_func(x, "conv4_1_1")
        x = self.conv_func(x, "conv4_1_2")
        x = self.conv_func(x, "conv4_1_3")
        # x = self.conv_func(x, "conv4_1_4")
        x = self.conv_func(x, "conv4_2_1")
        x = self.conv_func(x, "conv4_2_2")
        x = self.conv_func(x, "conv4_2_3")
        x = self.conv_func(x, "conv4_3_1")
        x = self.conv_func(x, "conv4_3_2")
        x = self.conv_func(x, "conv4_3_3")
        x = self.conv_func(x, "conv4_4_1")
        x = self.conv_func(x, "conv4_4_2")
        x = self.conv_func(x, "conv4_4_3")

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)

        total_diff = abs(self.total_real_time - self.total_predict_time)
        total_mape = total_diff / self.total_real_time
        print("总时间：" + str(self.total_real_time) + ", 预测：" + str(self.total_predict_time) + "差值：" +
              str(total_diff) + "MAPE: " + str(total_mape))

        return x

    def conv_func(self, x, name):
        # in_channel = x.shape[1]
        count = 30       # 当执行时间太短无法测量cpu时，会增加count，共执行count次
        pre_x = x
        real = 0
        predict = 0
        iteration = 1  # 循环20次，取平均值
        while True:
            if name == "conv0":
                start_time, start_cpu = getCpuAndTime(pid)
                for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                    x = self.conv0(pre_x)
                end_time, end_cpu = getCpuAndTime(pid)
            elif name == "conv1_1_1":
                start_time, start_cpu = getCpuAndTime(pid)
                for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                    x = self.conv1_1_1(pre_x)
                end_time, end_cpu = getCpuAndTime(pid)
            elif name == "conv1_1_2":
                start_time, start_cpu = getCpuAndTime(pid)
                for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                    x = self.conv1_1_2(pre_x)
                end_time, end_cpu = getCpuAndTime(pid)
            elif name == "conv1_1_3" or name == "conv1_1_4":
                start_time, start_cpu = getCpuAndTime(pid)
                for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                    x = self.conv1_1_3(pre_x)
                end_time, end_cpu = getCpuAndTime(pid)
            elif name == "conv1_2_1" or name == "conv1_3_1":
                start_time, start_cpu = getCpuAndTime(pid)
                for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                    x = self.conv1_2_1(pre_x)
                end_time, end_cpu = getCpuAndTime(pid)
            elif name == "conv1_2_2" or name == "conv1_3_2":
                start_time, start_cpu = getCpuAndTime(pid)
                for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                    x = self.conv1_2_2(pre_x)
                end_time, end_cpu = getCpuAndTime(pid)
            elif name == "conv1_2_3" or name == "conv1_3_3":
                start_time, start_cpu = getCpuAndTime(pid)
                for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                    x = self.conv1_2_3(pre_x)
                end_time, end_cpu = getCpuAndTime(pid)

            elif name == "conv2_1_1":
                start_time, start_cpu = getCpuAndTime(pid)
                for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                    x = self.conv2_1_1(pre_x)
                end_time, end_cpu = getCpuAndTime(pid)
            elif name == "conv2_1_2":
                start_time, start_cpu = getCpuAndTime(pid)
                for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                    x = self.conv2_1_2(pre_x)
                end_time, end_cpu = getCpuAndTime(pid)
            elif name == "conv2_1_3" or name == "conv2_1_4":
                start_time, start_cpu = getCpuAndTime(pid)
                for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                    x = self.conv2_1_3(pre_x)
                end_time, end_cpu = getCpuAndTime(pid)
            elif name == "conv2_2_1" or name == "conv2_3_1" or name == "conv2_4_1":
                start_time, start_cpu = getCpuAndTime(pid)
                for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                    x = self.conv2_2_1(pre_x)
                end_time, end_cpu = getCpuAndTime(pid)
            elif name == "conv2_2_2" or name == "conv2_3_2" or name == "conv2_4_2":
                start_time, start_cpu = getCpuAndTime(pid)
                for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                    x = self.conv2_2_2(pre_x)
                end_time, end_cpu = getCpuAndTime(pid)
            elif name == "conv2_2_3" or name == "conv2_3_3" or name == "conv2_4_3":
                start_time, start_cpu = getCpuAndTime(pid)
                for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                    x = self.conv2_2_3(pre_x)
                end_time, end_cpu = getCpuAndTime(pid)

            elif name == "conv3_1_1":
                start_time, start_cpu = getCpuAndTime(pid)
                for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                    x = self.conv3_1_1(pre_x)
                end_time, end_cpu = getCpuAndTime(pid)
            elif name == "conv3_1_2":
                start_time, start_cpu = getCpuAndTime(pid)
                for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                    x = self.conv3_1_2(pre_x)
                end_time, end_cpu = getCpuAndTime(pid)
            elif name == "conv3_1_3" or name == "conv3_1_4":
                start_time, start_cpu = getCpuAndTime(pid)
                for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                    x = self.conv3_1_3(pre_x)
                end_time, end_cpu = getCpuAndTime(pid)
            elif name == "conv3_2_1" or name == "conv3_3_1" or name == "conv3_4_1" or name == "conv3_5_1" or name == "conv3_6_1":
                start_time, start_cpu = getCpuAndTime(pid)
                for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                    x = self.conv3_2_1(pre_x)
                end_time, end_cpu = getCpuAndTime(pid)
            elif name == "conv3_2_2" or name == "conv3_3_2" or name == "conv3_4_2" or name == "conv3_5_2" or name == "conv3_6_2":
                start_time, start_cpu = getCpuAndTime(pid)
                for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                    x = self.conv3_2_2(pre_x)
                end_time, end_cpu = getCpuAndTime(pid)
            elif name == "conv3_2_3" or name == "conv3_3_3" or name == "conv3_4_3" or name == "conv3_5_3" or name == "conv3_6_3":
                start_time, start_cpu = getCpuAndTime(pid)
                for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                    x = self.conv3_2_3(pre_x)
                end_time, end_cpu = getCpuAndTime(pid)

            elif name == "conv4_1_1":
                start_time, start_cpu = getCpuAndTime(pid)
                for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                    x = self.conv4_1_1(pre_x)
                end_time, end_cpu = getCpuAndTime(pid)
            elif name == "conv4_1_2":
                start_time, start_cpu = getCpuAndTime(pid)
                for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                    x = self.conv4_1_2(pre_x)
                end_time, end_cpu = getCpuAndTime(pid)
            elif name == "conv4_1_3" or name == "conv4_1_4":
                start_time, start_cpu = getCpuAndTime(pid)
                for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                    x = self.conv4_1_3(pre_x)
                end_time, end_cpu = getCpuAndTime(pid)
            elif name == "conv4_2_1" or name == "conv4_3_1" or name == "conv4_4_1":
                start_time, start_cpu = getCpuAndTime(pid)
                for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                    x = self.conv4_2_1(pre_x)
                end_time, end_cpu = getCpuAndTime(pid)
            elif name == "conv4_2_2" or name == "conv4_3_2" or name == "conv4_4_2":
                start_time, start_cpu = getCpuAndTime(pid)
                for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                    x = self.conv4_2_2(pre_x)
                end_time, end_cpu = getCpuAndTime(pid)
            elif name == "conv4_2_3" or name == "conv4_3_3" or name == "conv4_4_3":
                start_time, start_cpu = getCpuAndTime(pid)
                for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                    x = self.conv4_2_3(pre_x)
                end_time, end_cpu = getCpuAndTime(pid)

            real_time = (end_time - start_time) / count * 1000
            cpu = round((end_cpu - start_cpu) / (end_time - start_time) * num_cpus * 100, 2)
            if cpu == 0 or cpu > 100:
                continue
            flops = get_conv_flops(pre_x.shape[1], x.shape[1], x.shape[2], 3)
            xy = np.vstack((np.array([flops]), np.array([cpu])))
            predict_time = Pfun(xy, *conv_popt)
            print(name + " ---  flops: " + str(flops) + "cpu: " + str(cpu) + "real: " + str(real_time) +
                  "predict: " + str(predict_time) + "diff: " + str(abs(real_time - predict_time)))
            real += real_time
            predict += predict_time
            break
        aver_real = real / iteration
        aver_predict = predict / iteration
        # print("平均-----总时间：" + str(aver_real) + ", 预测：" + str(aver_predict) + "差值：" +
        #       str(abs(aver_real - aver_predict)))
        self.total_real_time += aver_real
        self.total_predict_time += aver_predict
        return x

    def linear_func(self, x, name):
        count = 30
        pre_x = x
        real = 0
        predict = 0
        iteration = 1  # 循环20次，取平均值
        while True:
            if name == "linear1":
                start_time, start_cpu = getCpuAndTime(pid)
                for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                    x = self.linear1(pre_x)
                end_time, end_cpu = getCpuAndTime(pid)
            real_time = (end_time - start_time) / count * 1000
            cpu = round((end_cpu - start_cpu) / (end_time - start_time) * num_cpus * 100, 2)
            if cpu == 0 or cpu > 100:
                continue
            flops = get_linear_flops(pre_x.shape[1], x.shape[1])
            xy = np.vstack((np.array([flops]), np.array([cpu])))
            predict_time = Pfun(xy, *linear_popt)
            print(name + " ---  flops: " + str(flops) + "cpu: " + str(cpu) + "real: " + str(real_time) +
                  "predict: " + str(predict_time) + "diff: " + str(abs(real_time - predict_time)))
            real += real_time
            predict += predict_time
            break

        aver_real = real / iteration
        aver_predict = predict / iteration

        # print("平均-----总时间：" + str(aver_real) + ", 预测：" + str(aver_predict) + "差值：" +
        #       str(abs(aver_real - aver_predict)))
        self.total_real_time += aver_real
        self.total_predict_time += aver_predict
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
    df = pd.read_excel('datasets/conv_cpu_flops_130_4300.xls')
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
    df = pd.read_excel('datasets/linear_cpu_flops_3_260.xls')
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


def ResNet50():
    return ResNet([3, 4, 6, 3])


if __name__=='__main__':

    global pid
    pid = get_pid()
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

    model = ResNet50()
    input = torch.randn(1, 3, 224, 224)
    out = model(input)
