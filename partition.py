import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
# from PartitionTest import readJson, plotBar

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from On_demand_Fine_grained_Partitioning import readJson, plotBar

plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）

nodes = []
B = []
vims = []
net_name = ""

input_height = 227
input_c = 3

def get_mips(vim):
    mips = 0
    if vim == 0:
        mips = 1.0 / (2.81 * math.exp(-10))
    elif vim == 1:
        mips = 1.0 / (6.24 * math.exp(-11))
    elif vim == 2:
        mips = 1.0 / (5.04 * math.exp(-11))
    return mips


# 时间 = a * flops + b
def get_ab(vim, type):
    a = 0
    b = 0
    a_conv = [2.81e-10, 6.24e-11, 5.04e-11, 9.361e-12]
    b_conv = [1.37e-1,  1.97e-2,  5.83e-3,  2.726619e-9]

    a_fc = [3.09e-10, 5.12e-9, 4.82e-9, 1.64619e-10]
    b_fc = [4.11e-1,  8.28e-5, 5.73e-5, 1.83814e-10]

    if type == 'conv':
        a = a_conv[vim]
        b = b_conv[vim]
    elif type == 'fc':
        a = a_fc[vim]
        b = b_fc[vim]
    else:
        a = 0
        b = 0.0001
    return a, b
    # if vim == 0:
    #     if type == 'conv':
    #         a = 2.81e-10
    #         b = 1.37e-1
    #     elif type == 'fc':
    #         a = 3.09e-10
    #         b = 4.11e-1
    # elif vim == 1:
    #     if type == 'conv':
    #         a = 6.24e-11
    #         b = 1.97e-2
    #     elif type == 'fc':
    #         a = 5.12e-9
    #         b = 8.28e-5
    # elif vim == 2:
    #     if type == 'conv':
    #         a = 5.04e-11
    #         b = 5.83e-3
    #     elif type == 'fc':
    #         a = 4.82e-9
    #         b = 5.73e-5


def get_predict_time(vim, type, flops):
    a, b = get_ab(vim, type)
    return a * flops + b


def plot(times, xtricks, label):
    x = [i for i in range(len(times[0]))]
    move = 0.3
    width = 0.3
    # x_center = x
    for count in range(len(times)):
        plt.bar([i + move * count for i in x], times[count], width=width, label=label[count])  # 绘制柱状图
    plt.legend()
    plt.xticks(x, xtricks)
    plt.ylabel("时延")
    plt.xlabel("带宽")
    plt.show()


def allDevice(vim):
    # 全部在本地执行的时间
    time = 0
    for i in range(len(nodes)):
        time = time + get_predict_time(vim, nodes[i].type, nodes[i].flops)
    print("全部在本地执行的时间" + str(time))
    return time


def allEdge(vim, bandwidth):
    # 全部在边执行的时间
    time = nodes[0].height_in ** 2 * nodes[0].c_in * 4 / bandwidth  # 发送第一层
    for i in range(0, len(nodes)):  # 默认第0个设备是本地
        time = time + get_predict_time(vim, nodes[i].type, nodes[i].flops)
    print("带宽" + str(bandwidth) + "边缘" + str(vim) + "全部在边的执行时间" + str(time))
    return time


def minAllEdge(edge_vim_list, edge_b_list):
    minTime = float('inf')
    for i in range(len(edge_vim_list)):
        vim = edge_vim_list[i]
        b = edge_b_list[i]
        minTime = min(minTime, allEdge(vim, b))
    return minTime


def twoPart(localvim, edgevim, bandwidth):
    # 切成两部分
    minTime = float("inf")
    for partitionIndex in range(len(nodes)+1):
        localTime = 0
        edgeTime = 0
        # index = 0
        for index in range(partitionIndex):
            localTime = localTime + get_predict_time(localvim, nodes[index].type, nodes[index].flops)
        if partitionIndex < len(nodes):
            transTime = nodes[partitionIndex].height_in ** 2 * nodes[partitionIndex].c_in * 4 / bandwidth  # 发送最后一层
        for index in range(partitionIndex, len(nodes)):
            edgeTime = edgeTime + get_predict_time(edgevim, nodes[index].type, nodes[index].flops)
        time = localTime + transTime + edgeTime
        minTime = min(minTime, time)
    print("带宽:" + str(bandwidth) + "本地:" + str(localvim) + "边缘:" + str(edgevim) + "切成两部分的执行的时间:" + str(minTime))
    return minTime


def minTwoPart(local_vim, edge_vim_list, edge_b_list):
    minTime = float('inf')
    for i in range(len(edge_vim_list)):
        edge_vim = edge_vim_list[i]
        b = edge_b_list[i]
        minTime = min(minTime, twoPart(local_vim, edge_vim, b))
    return minTime


def hroizontal_partition():
    # MIPS_edge = MIPS[1:]
    vim_edge = vims[1:]
    B_edge = B[1:]
    datanode_num = len(vim_edge)
    hori_time = 0
    for i in range(len(nodes)):
        if nodes[i].type == 'conv':
            original_tensor = torch.randn(1, nodes[i].c_in, nodes[i].height_in, nodes[i].height_in)
            hori_time = hori_time + max(tensor_divide_by_computing_and_network(original_tensor, datanode_num=datanode_num,
                                                                           cross_layer=1,
                                                                           vims=vim_edge, B=B_edge, c_out=nodes[i].c_out, k=nodes[i].k_size))
        else:
            hori_time = hori_time + get_predict_time(0, nodes[i].type, nodes[i].flops)
    # print("执行时间" + str(hori_time))
    return hori_time


# 时间估计
def get_prediction_time(datanode_num = 0, index = 0, length = 0, cross_layer = 1, vims = vims, B = B, input_param = [], c_out = 0, k= 1):
    input_number, c_in, height, width = input_param
    if c_out == 0:
        c_out = c_in
    else:
        c_out = c_out
    kernel = k
    w_spread = (k-1) / 2
    # 计算 FLOPs
    FLOPs = 2 * height * length * c_out * (kernel * kernel * c_in + 1)
    # 计算时间
    a, b = get_ab(vims[index], 'conv')
    comp_time = a * FLOPs + b
    # 通信开销
    comm_data = input_number * c_in * height * 4 / B[index]
    comm_time = 0
    # 判断是否是边界
    if index == 0 or index == datanode_num - 1:
        comm_time = comm_data * (cross_layer * w_spread + 2 * length)      # 最后还要返回
    # 中间情况
    else:
        comm_time = comm_data * (2 * cross_layer * w_spread + 2 * length)
    prediction_time = comp_time + comm_time
    return prediction_time


# #############################################################################################################
# 根据计算能力划分区域, original_tensor默认为4维，划分后的[start, end],含start，不包含end
def tensor_divide_by_computing_and_network(original_tensor, datanode_num = 1, cross_layer = 1,
                                        vims = vims, B = B, c_out = 0, k = 1):
    # 优化步长
    step = 1
    divided_tensor = []
    divide_record = np.zeros((datanode_num, 2), dtype=int)
    input_param = []
    if datanode_num == 1:
        return original_tensor, divide_record
    else:
        input_number, c_in, height, width = original_tensor.size()
        input_param.append(input_number)
        input_param.append(c_in)
        input_param.append(height)
        input_param.append(width)
        # 提前计算求和
        total_computing_power = 0
        MIPS = []
        for i in range(datanode_num):
            total_computing_power += get_mips(vims[i])
            MIPS.append(total_computing_power)
        sum_computing_power = []
        for i in range(datanode_num + 1):
            sum_computing_power.append(sum(MIPS[0 : i]))

        # 定义划分长度
        length = []
        # 时间开销
        prediction_time = []
        for i in range(datanode_num):
            length.append(width // datanode_num)
            prediction_time.append(0)
        length[datanode_num-1] = width - (width // datanode_num) * datanode_num     # 如果划分不平均，多余的部分给最后一个节点
        # for it in range(datanode_num):
        #     length[it] = int(sum_computing_power[it+1]/total_computing_power * width) - \
        #                  int(sum_computing_power[it] / total_computing_power * width)
        for it in range(datanode_num):
            prediction_time[it] = get_prediction_time(datanode_num = datanode_num, index = it, length = length[it], cross_layer = cross_layer, vims = vims,
                        B = B, input_param=input_param, c_out = c_out, k = k)
        iter = 0
        iter_stop = 30
        diff = 0
        # 判断退出条件,1、max与min差值小于10ms，或者差值变化很小，或者某一个i对应的长度接近 1
        while(True):
            iter += 1
            # 判断是否到轮次上限
            if iter == iter_stop:
                break
            # 找出时间最值及下标
            max_value = max(prediction_time)
            min_value = min(prediction_time)
            index_max = prediction_time.index(max_value)
            index_min = prediction_time.index(min_value)
            last_diff = diff
            diff = max_value - min_value
            # 判断退出条件
            if ( diff < 0.02 or min(length) <= 2):
                break
            last_diff = diff
            length[index_max] -= step
            length[index_min] += step
            # 出错
            prediction_time[index_max] = get_prediction_time(datanode_num = datanode_num, index = index_max, length = length[index_max], cross_layer = 1,
                        vims = vims, B=B, input_param=input_param, c_out = c_out, k = k)
            prediction_time[index_min] = get_prediction_time(datanode_num = datanode_num, index = index_min, length = length[index_min], cross_layer = 1,
                        vims = vims, B=B, input_param=input_param, c_out = c_out, k= k)
        #     print (length)
        #     print (prediction_time)
        # print(length)
        # print(prediction_time)
        # 已经得到length，根据length确定划分范围
        start = 0
        end = 0
        for it in range(datanode_num):
            end = start + length[it]
            # print("[ %d, %d]" % (start, end))
            divide_record[it][0] = start
            divide_record[it][1] = end
            # 判断划分的位置
            temp_tensor = 0
            if it == 0:
                # 最左边划分
                temp_tensor = original_tensor[:, :, :, start : int(end + cross_layer)]
            elif it == datanode_num - 1:
                # 最右边划分
                temp_tensor = original_tensor[:, :, :, int(start - cross_layer) : end]
            else:
                # 中间非边界情况
                temp_tensor = original_tensor[:, :, :, int(start - cross_layer) : int(end + cross_layer)]
            # 放入list
            divided_tensor.append(temp_tensor)
            # 更换起始位置。
            start = end
    # 返回最终的结果
    # return divided_tensor, divide_record
    # print("水平" + str(prediction_time))
    return prediction_time


def pre_partition(name, input_height, input_c):
    global nodes, B, vims, net_name
    net_name = name
    nodes_all = readJson.constructByJson(net_name, input_height, input_c) # 构造网络
    nodes = nodes_all
    # for node in nodes_all:  # 只考虑卷积层
    #     if node.type == 'conv':
    #         nodes.append(node)

    B = [-1, 1000e6, 100e6, 1000e6, 1000e6]
    vims = [0, 2, 1, 2, 2]  # 0 树莓派  1 虚拟机1   2 虚拟机2,  默认第0个设备是本地
    # nodes_res = nodes
    return nodes


def partition():
    # cross_layer = 1

    global B, vims
    # # 高带宽 1000Mbps 不同设备数量对应的时延
    # B_temp = [-1, 1000e6, 1000e6, 1000e6, 1000e6, 1000e6]
    # vims_temp = [0, 1, 1, 1, 1, 1]  # 0 树莓派  1 虚拟机1   2 虚拟机2,  默认第0个设备是本地
    #
    # latency = []
    # bar_label = []
    # for i in range(3, len(B_temp)+1):
    #     B = B_temp[0:i]      # 设备数量 i-1
    #     vims = vims_temp[0:i]
    #     deviceTime = allDevice(vims[0]) * 1000
    #     edgeTime = minAllEdge(vims[1:], B[1:]) * 1000
    #     twoPartTime = minTwoPart(vims[0], vims[1:], B[1:]) * 1000
    #     horizontalTime = hroizontal_partition() * 1000
    #     time_list = [deviceTime, edgeTime, twoPartTime, horizontalTime]
    #     latency.append(time_list)
    #     bar_label.append(str(i-1))
    # x_ticks = ['local', 'edge', 'vertical', 'horizontal']
    # plotBar.plot(latency, bar_label, x_ticks)
    # for i in range(len(latency)):
    #     for j in range(len(latency[i])):
    #         latency[i][j] = format(latency[i][j], '.4f')
    # print(latency)


    ############# 相同设备数量4个， 同构设备， 不同带宽 50Mbps 100 500 1000时， 数量对应的时延   #############
    vims_temp = [0, 1, 1, 1]  # 0 树莓派  1 虚拟机1   2 虚拟机2,  默认第0个设备是本地
    B_temp = [100e6, 500e6, 1000e6]
    latency = []
    bar_labels = []
    for i in range(len(B_temp)):
        B = [-1, B_temp[i], B_temp[i], B_temp[i], B_temp[i]]
        vims = vims_temp
        deviceTime = allDevice(vims[0]) * 1000
        edgeTime = minAllEdge(vims[1:], B[1:]) * 1000
        # twoPartTime = minTwoPart(vims[0], vims[1:], B[1:]) * 1000
        horizontalTime = hroizontal_partition() * 1000
        time_list = [deviceTime, edgeTime, horizontalTime]
        latency.append(time_list)
        bar_labels.append(str(B_temp[i]/1000000) + 'Mbps')
    x_labels = ['local', 'edge', 'horizontal']
    # plotBar.plot(latency, bar_label, x_ticks)
    plotBar.create_multi_bars(latency, 'Latency', x_labels, 'latency(ms)', bar_labels)
    for i in range(len(latency)):
        for j in range(len(latency[i])):
            latency[i][j] = format(latency[i][j], '.4f')
    print(latency)


    # # 水平切割的时延随设备数量，网络带宽的变化,折线图
    # vims_temp = [0, 1, 1, 1, 1]  # 0 树莓派  1 虚拟机1   2 虚拟机2,  默认第0个设备是本地
    # B_temp = [50e6, 100e6, 500e6, 1000e6]
    # latency = []
    # zx_label = []
    # zx_xticks = []
    # for i in range(len(B_temp)):
    #     B_ttmp = [-1, B_temp[i], B_temp[i], B_temp[i], B_temp[i], B_temp[i]]
    #     latency_diff_num = []
    #     zx_xticks = []
    #     for j in range(3, len(B_ttmp)+1):       # 设备数量
    #         B = B_ttmp[0:j]
    #         vims = vims_temp[0:j]
    #         horizontalTime = hroizontal_partition() * 1000
    #         latency_diff_num.append(horizontalTime)
    #         zx_xticks.append(str(j-1))
    #     latency.append(latency_diff_num)
    #     zx_label.append(str(B_temp[i]/1000000) + 'Mbps')
    # x = [i for i in range(4)]
    # for i in range(len(latency)):
    #     plt.plot(x, latency[i], '-', label = zx_label[i])
    # print(latency)

    # vims_temp = [0, 1, 1, 2, 2]  # 0 树莓派  1 虚拟机1   2 虚拟机2,  默认第0个设备是本地
    # B_temp = [50e6, 100e6, 500e6, 1000e6]
    # latency = []
    # bar_label = []
    # for i in range(len(B_temp)):
    #     B = [-1, B_temp[i], B_temp[i], B_temp[i], B_temp[i]]
    #     vims = vims_temp
    #     deviceTime = allDevice(vims[0]) * 1000
    #     edgeTime = minAllEdge(vims[1:], B[1:]) * 1000
    #     twoPartTime = minTwoPart(vims[0], vims[1:], B[1:]) * 1000
    #     horizontalTime = hroizontal_partition() * 1000
    #     time_list = [deviceTime, edgeTime, twoPartTime, horizontalTime]
    #     latency.append(time_list)
    #     bar_label.append(str(B_temp[i] / 1000000) + 'Mbps')
    # x_ticks = ['local', 'edge', 'vertical', 'horizontal']
    # plotBar.plot(latency, bar_label, x_ticks)
    # # for i in range(len(latency)):
    # #     for j in range(len(latency[i])):
    # #         latency[i][j] = format(latency[i][j], '.4f')
    # print(latency)


def net_structure():
    outputSize = []
    # latency = []
    x_ticks = []
    # width = 0.4
    outputSize.append(nodes[0].in_size / 1000000)
    x_ticks.append('input')
    # latency.append(0)
    # vim = 1
    for i in range(len(nodes)):
        outputSize.append(nodes[i].out_size / 1000000)
        # latency.append(get_predict_time(vim, nodes[i].type, nodes[i].flops))
        x_ticks.append(nodes[i].name)
    x = np.arange(len(outputSize))
    # plt.figure(1)
    plt.bar(x, outputSize)
    # plt.bar(x+width, latency, width=width)
    plt.xticks(x + 0.1, x_ticks, rotation='vertical')
    plt.ylabel('Data size(MB)')
    plt.title(net_name)
    plt.show()

def get_name(nodes_tmp):
    names = ['input']
    for i in range(len(nodes_tmp)):
        names.append(nodes_tmp[i].name)
    return names

def net_latency():
    latency_list = []
    x_ticks = get_name(nodes)
    vims = [0, 1, 2, 3]
    for vim in vims:
        latency = [0]
        for i in range(len(nodes)):
            latency.append(get_predict_time(vim, nodes[i].type, nodes[i].flops))
        latency_list.append(latency)
    plotBar.create_multi_bars(latency_list, net_name, x_ticks, 'Latency(s)', ['0', '1', '2', '3'])


def get_speed_ratio(B_temp):
    global B, vims
    vims_temp = [0, 1, 1, 1]  # 0 树莓派  1 虚拟机1   2 虚拟机2,  默认第0个设备是本地
    device_time = allDevice(vims[0]) * 1000      # 全部在本地执行的时间
    ratio_local = []
    # ratio_edge = []
    for i in range(len(B_temp)):
        B = [-1, B_temp[i], B_temp[i], B_temp[i], B_temp[i]]
        vims = vims_temp
        horizontalTime = hroizontal_partition() * 1000
        # edge_time = allEdge(vims[1], B_temp[i])
        ratio_local.append(device_time / horizontalTime)
        # ratio_edge.append(edge_time / horizontalTime)
    print(ratio_local)
    return ratio_local

def draw_speed_ratio():
    """
    绘图：绘制不同网络，在不同带宽情况下，水平切割时延相对于本地执行时延的加速比
    """
    B_temp = [50e6, 100e6, 200e6, 400e6, 600e6, 800e6, 1000e6]     # 带宽 x轴
    x = [i / 1000000 for i in B_temp]   # x轴
    pre_partition("AlexNet", 227, input_c)      # 注意：nodes使用的是全局变量
    ratio = get_speed_ratio(B_temp)
    plt.subplot(221)
    plt.plot(x, ratio, marker='o')
    plt.title('AlexNet')

    pre_partition("GoogleNet", 227, input_c)      # 注意：nodes使用的是全局变量
    ratio = get_speed_ratio(B_temp)
    plt.subplot(222)
    plt.plot(x, ratio, label='GoogleNet',color='r', marker='o')
    plt.title('GoogleNet')

    pre_partition("Vgg16", 224, input_c)  # 注意：nodes使用的是全局变量
    ratio = get_speed_ratio(B_temp)
    plt.subplot(223)
    plt.plot(x, ratio, label='Vgg16', color='g', marker='o')
    plt.title('Vgg16')

    pre_partition("ResNet50", 224, input_c)  # 注意：nodes使用的是全局变量
    ratio = get_speed_ratio(B_temp)
    plt.subplot(224)
    plt.plot(x, ratio, label='ResNet50', color='y', marker='o')
    plt.title('ResNet50')


if __name__ == '__main__':
    # pre_partition("GoogleNet", input_height, input_c)
    pre_partition("AlexNet", input_height, input_c)
    # pre_partition("Vgg16", input_height, input_c)
    # net_structure()
    # net_latency()
    partition()
    # draw_speed_ratio()
