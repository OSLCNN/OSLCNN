from datetime import datetime
import torch
import time
import torch.nn as nn
import pandas as pd
import os
import psutil
import sys

cpu_target = float(sys.argv[1])
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_timer = getattr(time, 'monotonic', time.time)  # 获取对象time的monotonic属性，如果属性不存在，默认值为time.time
num_cpus = psutil.cpu_count() or 1


def timer():
    return _timer()


# 获取当前进程PID
def get_pid():
    f = os.popen("pidof python GenerateData.py").read().split(' ')[0]
    print(f)
    return int(f)


def getCpuAndTime(pid):
    # 获取当前时间和cpu利用率
    p = psutil.Process(pid)
    pt = p.cpu_times()
    st1, pt1_0, pt1_1 = timer(), pt.user, pt.system  # new
    pt1 = pt1_0 + pt1_1
    return st1, pt1


# 只包含一层卷积层的网络
class Conv(nn.Module):
    def __init__(self, input_size=224, output_size=224, kernel_size=3, in_channel=3, out_channel=64, stride=1,
                 padding=1, init_weights=True):
        super(Conv, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
        )
        self.flops = 2 * output_size * output_size * (in_channel * kernel_size * kernel_size + 1) * out_channel
        if init_weights:
            self._initialize_weights()

    def get_flops(self):
        return self.flops

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Fully_layer(nn.Module):
    def __init__(self, input_size = 4096, output_size = 4096, init_weights = True ):
        super(Fully_layer, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(True),
        )
        self.flops = (2 * input_size - 1) * output_size
        if init_weights:
            self._initialize_weights()
    def get_flops(self):
        return self.flops

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def generate_conv():
    # 二维数据记录FLOPS，cpu利用率，与其对应的运行时间
    # 得到当前进程的进程ID
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

    # channel_list = [[3, 64], [64, 64], [64, 128], [128, 128], [128, 256], [256, 256], [256, 512]]
    channel_list = [[128, 128]]
    kernel_size = 3
    stride = 1
    padding = 1
    df = pd.DataFrame(columns=("input_size", "output_size", "in_channels", "out_channels", "kernel_size", "stride",
                               "padding", "flops", "cpu", "latency"))
    # 调节模型的参数(in_chanel, out_chanel, width)，得到新的模型
    row = 0
    for channel in channel_list:
        in_channel = channel[0]
        out_channel = channel[1]
        print("通道数:" + str(channel))
        for output_size in range(18, 128, 3):
            # 模型的输入
            input_size = output_size
            input = torch.randn(1, in_channel, input_size, input_size)
            model = Conv(input_size=input_size, output_size=output_size, kernel_size=kernel_size,
                         in_channel=in_channel,
                         out_channel=out_channel, stride=stride, padding=padding, init_weights=True)
            counts = 1
            print("input_size: " + str(input_size) + "flops: " + str(model.get_flops() / (10 ** 6)))
            # 运行count次，取最接近cpu_target的
            count = 5
            cpu_optimal = 0
            latency_optimal = 0

            for i in range(count):

                start_time, start_cpu = getCpuAndTime(pid)
                # 模型开始计算
                for i in range(counts):
                    output = model(input)
                end_time, end_cpu = getCpuAndTime(pid)

                # 单位分别是：M， 和 ms（便于运算）
                # 根据输入和conv层参数计算FLOPs值
                flops = model.get_flops() / (10 ** 6)

                delta_proc = end_cpu - start_cpu
                delta_time = end_time - start_time
                latency = delta_time / counts * 1000
                try:
                    cpus_percent = ((delta_proc / delta_time * num_cpus) * 100)  # cpu利用率
                except:
                    cpus_percent = 0.0
                count = 30
                while cpus_percent == 0.0 or cpus_percent > 100:
                    start_time, start_cpu = getCpuAndTime(pid)
                    for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                        output = model(input)
                    end_time, end_cpu = getCpuAndTime(pid)
                    delta_proc = end_cpu - start_cpu
                    delta_time = end_time - start_time
                    latency = delta_time / count * 1000
                    try:
                        cpus_percent = ((delta_proc / delta_time * num_cpus) * 100)
                    except:
                        cpus_percent = 0.0
                    count = count + 30

                if abs(cpus_percent - cpu_target) < abs(cpu_optimal - cpu_target):
                    cpu_optimal = cpus_percent
                    latency_optimal = latency

            df.loc[row] = [input_size, output_size, in_channel, out_channel, kernel_size, stride, padding, flops,
                           100.0 - cpu_optimal, latency_optimal]
            row = row + 1
            df.to_excel('datasets/conv_cpu_flops_time.xls', sheet_name='conv_cpu_flops_time')
        print("---------------测试完成一组output_size数据---------------")


def generate_fc():
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

    # channel_list = [[3, 64], [64, 64], [64, 128], [128, 128], [128, 256], [256, 256], [256, 512]]
    channel_list = [[512, 512]]
    df = pd.DataFrame(columns=("input_size", "output_size", "flops", "cpu", "latency"))
    # 调节模型的参数(in_chanel, out_chanel, width)，得到新的模型
    row = 0

    for input_size in range(1000, 11696, 200):
        output_size = input_size
        input = torch.randn(input_size)
        model = Fully_layer(input_size=input_size, output_size=output_size, init_weights=True)
        print("input_size:" + str(input_size) + "output_size:" + str(output_size) +
              "flops:" + str(model.get_flops() / (10 ** 6)))
        counts = 1
        # 运行count次，取最接近cpu_target的
        count = 5
        cpu_optimal = 0
        latency_optimal = 0
        for i in range(count):
            start_time, start_cpu = getCpuAndTime(pid)
            # 模型开始计算
            for k in range(counts):
                output = model(input)
            end_time, end_cpu = getCpuAndTime(pid)
            # 单位分别是：M， 和 ms（便于运算）
            # 根据输入和conv层参数计算FLOPs值
            flops = model.get_flops() / (10 ** 6)

            delta_proc = end_cpu - start_cpu
            delta_time = end_time - start_time
            latency = delta_time / counts * 1000
            try:
                cpus_percent = ((delta_proc / delta_time * num_cpus) * 100)  # cpu利用率
            except:
                cpus_percent = 0.0
            count = 30
            while cpus_percent == 0.0 or cpus_percent > 100:
                start_time, start_cpu = getCpuAndTime(pid)
                for j in range(count):  # 执行时间太太太短了，多执行count次后取平均
                    output = model(input)
                end_time, end_cpu = getCpuAndTime(pid)

                delta_proc = end_cpu - start_cpu
                delta_time = end_time - start_time
                latency = delta_time / count * 1000
                try:
                    cpus_percent = ((delta_proc / delta_time * num_cpus) * 100)
                except:
                    cpus_percent = 0.0
                count = count + 30
            if abs(cpus_percent - cpu_target) < abs(cpu_optimal - cpu_target):
                cpu_optimal = cpus_percent
                latency_optimal = latency

        df.loc[row] = [input_size, output_size, flops, 100.0 - cpu_optimal, latency_optimal]

        row = row + 1
        df.to_excel('datasets/linear_cpu_flops_time.xls', sheet_name='conv_cpu_flops_time')
        print("测试完成一组数据" + str(cpu_optimal) + ", count:" + str(count))
    print("---------------测试完成一组output_size数据---------------")


if __name__ == '__main__':
    generate_conv()
