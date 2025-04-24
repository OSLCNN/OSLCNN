from typing import List

from On_demand_Fine_grained_Partitioning import Util, parameters, multiPartition
from On_demand_Fine_grained_Partitioning.Node import Node, ConvNode, CombineNode
from On_demand_Fine_grained_Partitioning.Device import Device
from On_demand_Fine_grained_Partitioning.Util import get_series_flops, predict_time, get_transmission_time


def get_conv_time(node, center_device, edge_devices, height_in, width_in, c_in, height_out, width_out, c_out):
    # 计算水平分割后，卷积层的时间
    device_num = len(edge_devices)
    step = parameters.step  # 优化步长
    if device_num == 1:
        # 不需要分割，直接发送完整的卷积层
        node.vmId = edge_devices[0].id
        time = predict_time(height_out, width_in, c_in, width_out, c_out, center_device, center_device, node, 0)
    else:
        length = []     # 定义划分长度(根据输出大小平均分)
        prediction_time = []    # 时间开销
        for i in range(device_num):
            length.append(height_out // device_num)
            prediction_time.append(0)
        # 如果划分不平均，多余的部分给最后一个节点
        length[device_num - 1] = height_out - (height_out // device_num) * (device_num-1)

        for it in range(device_num):
            node.vmId = edge_devices[it].id
            if it != 0:
                node.is_first = False
            else:
                node.is_first = True
            if it != device_num-1:
                node.is_last = False
            else:
                node.is_last = True
            prediction_time[it] = predict_time(length[it], width_in, c_in, width_out, c_out, center_device,
                                               center_device, node, 0)

        # 恢复
        node.is_first = True
        node.is_last = True

        iter = 0
        iter_stop = 30
        while True:  # 判断退出条件,1、max与min差值小于10ms，或者差值变化很小，或者某一个i对应的长度接近 1
            iter = iter + 1
            if iter == iter_stop:  # 判断是否到轮次上限
                break
            max_value = max(prediction_time)        # 找出时间最值及下标
            min_value = min(prediction_time)
            index_max = prediction_time.index(max_value)
            index_min = prediction_time.index(min_value)
            diff = max_value - min_value
            if diff < 20 or min(length) <= 2:   # 判断退出条件
                break
            length[index_max] -= step
            length[index_min] += step
            prediction_time[index_max] = predict_time(length[index_max], width_in, c_in, width_out, c_out,
                                                      center_device, center_device, node, 0)
            prediction_time[index_min] = predict_time(length[index_min], width_in, c_in, width_out, c_out,
                                                      center_device, center_device, node, 0)
        for it in range(device_num):
            node.vmId = edge_devices[it].id
            if it != 0:
                node.is_first = False
            else:
                node.is_first = True
            if it != device_num-1:
                node.is_last = False
            else:
                node.is_last = True
            prediction_time[it] = predict_time(length[it], width_in, c_in, width_out, c_out, center_device,
                                               center_device, node)

        time = max(prediction_time)
    return time


def partition(nodes: List[Node], devices: List[Device], B):

    # 设置带宽
    Util.B = B
    Util.devices = devices
    # edge_devices = []
    # for device in devices:
    #     if device.type == 1:
    #         edge_devices.append(device)
    # # 选择最强的边缘设备（最少一个，最多三个）
    # if len(edge_devices) < 1:
    #     print("边缘设备数量少于1")
    #     return -1
    # elif len(edge_devices) >= 3:
    #     edge_devices = edge_devices[len(edge_devices)-3 : len(edge_devices)]

    # 只有一个设备
    if len(devices) == 1:
        return multiPartition.no_partition(devices[0], cpu)

    # 选择最强的边缘设备（最少一个，最多三个）
    # if len(devices) < 3:
    #     edge_devices = devices[1:]
    # elif len(devices) >= 3:
    #     edge_devices = devices[len(devices)-3 : len(devices)]

    # edge_devices = devices[1: len(devices)]
    edge_devices = devices[1: len(devices)]


    native_device = devices[0]  # 本地设备
    # center_device = edge_devices[len(edge_devices)-1]   # 选最强的那个设备作为中心设备
    center_device = native_device  # 选本地设备作为中心设备

    time_sum = 0
    # 将输入发送到center_device的传输时间
    input_node = nodes[0]
    child_nodes = input_node.child_nodes[0]
    # if center_device != native_device:
    #     time_sum += get_transmission_time(child_nodes[1], child_nodes[2], child_nodes[3], native_device, center_device)
    nodes = nodes[1:]   # 第一个元素为input节点，没有父节点
    for node in nodes:
        combine_nodes = [node]
        if len(node.combine_nodes) > 0:
            combine_nodes = node.combine_nodes
        parent_nodes = node.parent_nodes
        height_in = parent_nodes[0][1]
        width_in = parent_nodes[0][2]
        c_in = 0
        for parent in parent_nodes:
            c_in += parent[3]
        if isinstance(node, ConvNode) or (isinstance(node, CombineNode) and node.hasConv):
            child_nodes = node.child_nodes
            height_out = child_nodes[0][1]
            width_out = child_nodes[0][2]
            c_out = child_nodes[0][3]
            time = get_conv_time(node, center_device, edge_devices, height_in, width_in, c_in, height_out,
                                 width_out, c_out)
        else:
            # flops = get_series_flops(combine_nodes, height_in, width_in, c_in, True, True)
            time = 0
            for node_tmp in combine_nodes:
                height_out, width_out, c_out = Util.get_output_shape(node_tmp, height_in, width_in, c_in, True, True)
                flops = Util.get_flops(node_tmp, height_in, width_in, c_in, True, True, height_out, width_out, c_out)
                height_in = height_out
                width_in = width_out
                c_in = c_out
                time += native_device.predict_time_by_node(node_tmp, flops)
        time_sum += time
    return time_sum



