import copy
import json
import math
from math import exp, log, pi, atan
import pandas as pd
import numpy as np

from On_demand_Fine_grained_Partitioning import parameters
from On_demand_Fine_grained_Partitioning.Node import CombineNode, ConvNode, FcNode, PoolNode, ConvConcatNode, ConcatNode

"""
计算适应度值
"""

nodes = []
devices = []
B = []
# cpu = 0


def get_fitness(algorithm_name, nodes_tmp, decision):
    # flag:是否按需分配
    global nodes, devices, B
    nodes = copy.deepcopy(nodes_tmp)
    # devices = devices_tmp
    # B = B_tmp
    if algorithm_name == 'OFP':
        on_demand_assignment(decision)  # 按需分配卷积层
    elif algorithm_name == 'FP':
        combine_with_device(decision)       # 按照设备组织节点
    total_time = get_time(algorithm_name, decision)
    return 1 / total_time


def combine_with_device(decision):
    decision = np.insert(decision, 0, 0)
    for i, begin_node in enumerate(nodes):

        if begin_node.need_assign:  # 根据首结点来确定需要划分的节点的组合

            begin_node_child_nodes = begin_node.child_nodes
            index = i
            child_i = 0

            while child_i < len(begin_node_child_nodes):
                child_node = begin_node_child_nodes[child_i][0]
                end_node = child_node.child_nodes[0][0]
                if not isinstance(child_node, ConvConcatNode):
                    index += 1
                    child_i += 1
                    continue
                else:
                    nodes_tmp = []
                    device_id_visit_inner = []
                    height_out_sum = 0
                    for j in range(parameters.slice_num):
                        index += 1
                        device_id = decision[index]
                        current_node = nodes[index]
                        height_out_sum += current_node.child_nodes[0][1]
                        if device_id not in device_id_visit_inner:
                            device_id_visit_inner.append(device_id)
                            nodes_tmp.append(current_node)
                        else:
                            current_node.visible = False

                    real_slice_num = len(nodes_tmp)
                    nodes_tmp[len(nodes_tmp) - 1].is_last = True
                    for conv_concat_node in nodes_tmp:
                        height_out = height_out_sum // real_slice_num
                        real_slice_num -= 1
                        height_out_sum -= height_out
                        conv_concat_node.child_nodes[0][1] = height_out
                        for node_list in end_node.parent_nodes:
                            if node_list[0] == conv_concat_node:
                                node_list[1] = height_out
                                break
                        combine_nodes = [conv_concat_node]
                        if len(conv_concat_node.combine_nodes) > 0:
                            combine_nodes = conv_concat_node.combine_nodes
                        height_in = get_series_height_in(combine_nodes, height_out, conv_concat_node.is_first, conv_concat_node.is_last)
                        conv_concat_node.parent_nodes[0][1] = height_in
                        for node_list in begin_node.child_nodes:
                            if node_list[0] == conv_concat_node:
                                node_list[1] = height_in
                                break
                    child_i += parameters.slice_num


# 根据决策，决定分配方案
# 最新版
# #############################################################################################################
def on_demand_assignment(decision):
    decision = np.insert(decision, 0, 0)

    # decision 和 nodes 的顺序是一致的
    for i, begin_node in enumerate(nodes):

        if begin_node.need_assign:  # 根据首结点来确定需要划分的节点的组合

            # begin_device = devices[decision[i]]
            begin_node_child_nodes = begin_node.child_nodes
            begin_node.vmId = decision[i]

            branch_nodes = []  # 每个分支对应几个节点
            index = i
            child_i = 0
            device_id_visit = []  # 设备号，对应于device_nodes列表中的每一行

            while child_i < len(begin_node_child_nodes):
                child_node = begin_node_child_nodes[child_i][0]
                if not isinstance(child_node, ConvConcatNode):
                    index += 1
                    child_node.vmId = decision[index]
                    if decision[index] not in device_id_visit:
                        device_id_visit.append(decision[index])
                    branch_nodes.append([child_node])  # 不是分割后的节点，说明这个分支只有这一个节点
                    child_i += 1
                else:
                    nodes_tmp = []
                    device_id_visit_inner = []

                    for j in range(parameters.slice_num):
                        index += 1
                        device_id = decision[index]

                        if device_id not in device_id_visit:
                            device_id_visit.append(device_id)

                        current_node = nodes[index]
                        if device_id not in device_id_visit_inner:
                            current_node.vmId = device_id
                            device_id_visit_inner.append(device_id)
                            nodes_tmp.append(current_node)
                        else:
                            current_node.visible = False
                    nodes_tmp[len(nodes_tmp) - 1].is_last = True
                    branch_nodes.append(nodes_tmp)
                    child_i += parameters.slice_num

            # 下一个元素就是end_node
            end_node = nodes[index + 1]
            end_node.vmId = decision[index + 1]
            # end_device = devices[decision[index]]

            # 先根据输出height平均分配
            branch_height_out_arr = []  # 每个分支总的输出的height
            branch_width_out_arr = []
            branch_c_out_arr = []

            branch_height_in_arr = []  # 每个分支总的输入的height
            branch_width_in_arr = []
            branch_c_in_arr = []

            # 将集合根据设备进行组合
            branch_num = len(branch_nodes)  # 分支数
            device_num = len(device_id_visit)
            # 二维列表 第i行j列的元素表示第j个分支的节点在第i个设备上执行，如果为None表示该分支不在这个设备上执行
            device_nodes = [[None for col in range(branch_num)] for row in range(device_num)]
            branch_index = 0  # 第几个分支
            for branch_nodes_tmp in branch_nodes:
                branch_height_out_arr.append(branch_nodes_tmp[0].height_out)
                branch_width_out_arr.append(branch_nodes_tmp[0].width_out)
                branch_c_out_arr.append(branch_nodes_tmp[0].c_out)

                branch_height_in_arr.append(branch_nodes_tmp[0].height_in)
                branch_width_in_arr.append(branch_nodes_tmp[0].width_in)
                branch_c_in_arr.append(branch_nodes_tmp[0].c_in)

                for node in branch_nodes_tmp:
                    device_id = node.vmId
                    device_nodes[device_id_visit.index(device_id)][branch_index] = node
                branch_index += 1

            length = assign(branch_height_out_arr, branch_width_out_arr, branch_c_out_arr, branch_width_in_arr,
                            branch_c_in_arr, begin_node, end_node, device_nodes, branch_nodes)

            # 将输出转换为输入
            for device_index in range(device_num):
                for branch_index in range(branch_num):
                    node = device_nodes[device_index][branch_index]
                    if not node:
                        continue
                    height_out = length[device_index][branch_index]
                    combine_nodes = [node]
                    if len(node.combine_nodes) > 0:
                        combine_nodes = node.combine_nodes
                    height_in = get_series_height_in(combine_nodes, height_out, node.is_first, node.is_last)

                    # 修改node的parent和child的height
                    node.parent_nodes[0][1] = height_in
                    for child in begin_node.child_nodes:
                        if child[0] == node:
                            child[1] = height_in
                            break
                    node.child_nodes[0][1] = height_out
                    for parent in end_node.parent_nodes:
                        if parent[0] == node:
                            parent[1] = height_out


##########################################################
def predict_time(height_out, width_in, c_in, width_out, c_out, begin_device, end_device, between_node, cpu=-1):
    # is_first， is_last 表示这是否是第一个或最后一个切片，计算时需要增加一个padding，
    # 先根据输出大小，计算输入大小
    combine_nodes = [between_node]
    if len(between_node.combine_nodes) > 0:
        combine_nodes = between_node.combine_nodes

    height_in = get_series_height_in(combine_nodes, height_out, between_node.is_first, between_node.is_last)
    flops = get_series_flops(combine_nodes, height_in, width_in, c_in, between_node.is_first, between_node.is_last)

    between_device = devices[between_node.vmId]
    begin_trans_time = get_transmission_time(height_in, width_in, c_in, begin_device, between_device)
    # comp_time = flops / between_device.p / 1000 / 1000
    comp_time = between_device.predict_time_by_type('conv', flops, cpu)

    end_trans_time = get_transmission_time(height_out, width_out, c_out, between_device, end_device)
    return begin_trans_time + comp_time + end_trans_time


##########################################################
# 分配
def assign(branch_height_out_arr, branch_width_out_arr, branch_c_out_arr, branch_width_in_arr, branch_c_in_arr,
           begin_node, end_node, device_nodes, branch_nodes):
    device_num = len(device_nodes)  # 设备数量
    branch_num = len(device_nodes[0])  # 分支数

    length = [[0 for col in range(branch_num)] for row in range(device_num)]  # 定义划分长度(根据输出大小平均分)
    prediction_time = [[0 for col in range(branch_num)] for row in range(device_num)]  # 预测的时间，对应length

    for j in range(branch_num):  # 按列分配length
        current_branch_node_num = len(branch_nodes[j])  # 当前分支的节点数
        current_branch_height_out = branch_height_out_arr[j]  # 当前分支的输出大小
        current_branch_node_count = 0  # 遍历到的第几个节点
        for i in range(device_num):
            if device_nodes[i][j]:
                current_branch_node_count += 1
                if current_branch_node_count == current_branch_node_num:
                    # 如果划分不平均，多余的部分给最后一个节点
                    length[i][j] = current_branch_height_out - (current_branch_height_out // current_branch_node_num) * \
                                   (current_branch_node_num - 1)
                else:
                    length[i][j] = current_branch_height_out // current_branch_node_num
                # 根据分配的length初始化预测时间
                prediction_time[i][j] = predict_time(length[i][j], branch_width_in_arr[j], branch_c_in_arr[j],
                                                     branch_width_out_arr[j], branch_c_out_arr[j],
                                                     devices[begin_node.vmId], devices[end_node.vmId],
                                                     device_nodes[i][j])
            else:
                length[i][j] = 0  # 如果为None，表示分支j上没有在设备i上执行的节点，不需要分配长度
                prediction_time[i][j] = 0
    step = parameters.step  # 优化步长
    iter = 0
    iter_stop = 30
    while True:  # 判断退出条件,1、max与min差值小于10ms，或者差值变化很小，或者某一个i对应的长度接近 1
        iter = iter + 1
        if iter == iter_stop:  # 判断是否到轮次上限
            break
        prediction_time_sum = [sum(time) for time in prediction_time]  # 每个设备的执行总时间
        prediction_time_index = [i for i in range(device_num)]
        # 对时间从小到大排序，prediction_time_index为对应的索引
        for i in range(1, device_num):
            for j in range(0, len(prediction_time_sum) - i):
                if prediction_time_sum[j] > prediction_time_sum[j + 1]:
                    prediction_time_sum[j], prediction_time_sum[j + 1] = prediction_time_sum[j + 1], \
                                                                         prediction_time_sum[j]
                    prediction_time_index[j], prediction_time_index[j + 1] = prediction_time_index[j + 1], \
                                                                             prediction_time_index[j]

        max_value = prediction_time_sum[device_num - 1]  # 找出时间最值及下标
        min_value = prediction_time_sum[0]
        max_device_index = prediction_time_index[device_num - 1]
        min_device_index = prediction_time_index[0]
        diff = max_value - min_value
        if diff < 20:  # 判断退出条件
            break

        # print("time :" + str(prediction_time))
        # print("sum_sort : " + str(prediction_time_sum))
        # print("sum_sort_index : " + str(prediction_time_index))
        # print("max_device_index :" + str(max_device_index) + "min_device_index : " + str(min_device_index))

        # 从小到大，找最小的设备，该设备和时延最大的设备执行相同的分支
        for i in range(device_num - 1):
            min_value = prediction_time_sum[i]
            min_device_index = prediction_time_index[i]
            # 判断两个设备上，是否执行相同的分支, 如果有，得到差值最大的分支
            max_device_time_arr = prediction_time[max_device_index]
            min_device_time_arr = prediction_time[min_device_index]
            # max_diff_inner = 0
            max_diff_inner = -2333333
            max_branch_index = -1
            for j in range(branch_num):
                if max_device_time_arr[j] != 0 and min_device_time_arr[j] != 0:
                    diff_inner = max_device_time_arr[j] - min_device_time_arr[j]
                    if diff_inner > max_diff_inner:
                        max_diff_inner = diff_inner
                        max_branch_index = j
            if max_branch_index != -1:
                length[max_device_index][max_branch_index] -= step
                length[min_device_index][max_branch_index] += step
                break
            else:
                continue

        if max_branch_index == -1:  # 没有可交换的了
            break

        # print("max_device_index :" + str(max_device_index) + "min_device_index : " + str(min_device_index)
        #       + "max_branch_index : " + str(max_branch_index))

        prediction_time[max_device_index][max_branch_index] = predict_time(
            length[max_device_index][max_branch_index],
            branch_width_in_arr[max_branch_index], branch_c_in_arr[max_branch_index],
            branch_width_out_arr[max_branch_index],
            branch_c_out_arr[max_branch_index], devices[begin_node.vmId], devices[end_node.vmId],
            device_nodes[max_device_index][max_branch_index])

        prediction_time[min_device_index][max_branch_index] = predict_time(
            length[min_device_index][max_branch_index],
            branch_width_in_arr[max_branch_index], branch_c_in_arr[max_branch_index],
            branch_width_out_arr[max_branch_index],
            branch_c_out_arr[max_branch_index], devices[begin_node.vmId], devices[end_node.vmId],
            device_nodes[min_device_index][max_branch_index])

    return length


# 根据输入大小 以及 node节点，计算节点的flops
def get_flops(node, height_in, width_in, c_in, is_first, is_last, height_out=1, width_out=1, c_out=1):
    height_out, width_out, c_out = get_output_shape(node, height_in, width_in, c_in, is_first, is_last)
    width_in = width_in + 2 * node.padding
    if is_first:
        height_in = height_in + node.padding
    if is_last:
        height_in = height_in + node.padding
    if isinstance(node, ConvNode) or isinstance(node, ConvConcatNode):
        flops = 2.0 * height_in * width_in * (c_in * node.k_size * node.k_size + 1) * node.k_num
    elif isinstance(node, FcNode):
        flops = (2 * height_in * width_in * c_in - 1) * node.c_out
    elif isinstance(node, PoolNode):
        flops = node.k_size * node.k_size * height_out * width_out * c_in
    else:
        flops = 0
    return flops


def get_series_flops(combine_nodes, height_in, width_in, c_in, is_first, is_last):
    flops = 0
    for node in combine_nodes:
        height_out, width_out, c_out = get_output_shape(node, height_in, width_in, c_in, is_first, is_last)
        flops += get_flops(node, height_in, width_in, c_in, is_first, is_last, height_out, width_out, c_out)
        height_in = height_out
        width_in = width_out
        c_in = c_out
    return flops


def get_height_in(node, height_out, is_first, is_last):  # 根据输出的height计算输入的height
    height_in = height_out
    if isinstance(node, ConvNode) or isinstance(node, PoolNode) or (isinstance(node, ConvConcatNode) and node.hasConv):
        height_in = (height_out - 1) * node.stride + node.k_size
    if is_first:
        height_in -= node.padding
    if is_last:
        height_in -= node.padding
    return height_in


# 连续有多个节点，根据最后的输出，计算最开始的输入height
def get_series_height_in(combine_nodes, height_out, is_first, is_last):
    height_in = height_out
    for node in reversed(combine_nodes):
        height_in = get_height_in(node, height_in, is_first, is_last)
    return height_in


# 计算node节点的输出形状
def get_output_shape(node, height_in, width_in, c_in, is_first, is_last):  # is_first是第一个切片，is_last是最后一个切片
    if isinstance(node, ConvNode) or isinstance(node, PoolNode):
        width_in += 2 * node.padding
        if is_first:
            height_in += node.padding
        if is_last:
            height_in += node.padding
        height_out = math.ceil((height_in - node.k_size) / node.stride + 1)
        width_out = math.ceil((width_in - node.k_size) / node.stride + 1)
        if isinstance(node, ConvNode):
            c_out = node.k_num
        else:
            c_out = c_in
    elif isinstance(node, FcNode):
        height_out = width_out = 1
        c_out = node.out
    else:
        height_out, width_out, c_out = height_in, width_in, c_in
    return height_out, width_out, c_out


def get_series_out_shape(combine_nodes, height_in, width_in, c_in, is_first, is_last):
    height_out, width_out, c_out = height_in, width_in, c_in
    for node in combine_nodes:
        height_out, width_out, c_out = get_output_shape(node, height_out, width_out, c_out, is_first, is_last)
    return height_out, width_out, c_out


def get_transmission_time(height, width, c, begin_device, end_device):
    # 从 beigin_device 传输 大小为size的数据 到 end_device的时间
    if begin_device == end_device:
        return 0
    size = height * width * c * 4
    transmission_time = size / B[begin_device.type][end_device.type] / 8 / 1024 / 1024 * 1000
    return transmission_time


def get_time(algorithm_name, decision):  # 本地只有一个：device0
    # decision = [1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5]
    # df = pd.DataFrame(columns=['id', 'name', 'vmId', 'parent', 'children', 'k_size', 'k_num', 'stride', 'padding',
    #                            'flops', 'begin_time', 'end_time'])

    decision = np.insert(decision, 0, 0)  # 不要直接修改decision.insert(0,0)因为decision是数组，会直接修改地址

    # 开始之前，需要先重置设备的时间段
    for device in devices:
        device.period = []
    for i, j in enumerate(decision):

        node = nodes[i]
        node.beginTime = 0
        device = devices[j]
        nodes[i].vmId = j
        if not node.visible:
            continue

        # c_in = height_in = width_in = 0
        # for parent_list in node.parent_nodes:
        #     height_in = parent_list[1]
        #     width_in = parent_list[2]
        #     c_in += parent_list[3]

        c_in = height_in = width_in = 0
        parent_i = 0
        while parent_i < len(node.parent_nodes):
            parent_list = node.parent_nodes[parent_i]
            parent_node = parent_list[0]
            if not isinstance(parent_node, ConvConcatNode):
                c_in += parent_list[3]
                height_in = parent_list[1]
                width_in = parent_list[2]
                parent_i += 1
            else:
                inner_height_in = 0
                for r in range(parameters.slice_num):
                    if node.parent_nodes[parent_i+r][0].visible:
                        inner_height_in += node.parent_nodes[parent_i+r][1]
                height_in = inner_height_in
                width_in = parent_list[2]
                c_in += parent_list[3]
                parent_i += parameters.slice_num

        if len(node.combine_nodes) > 0:
            flops = get_series_flops(node.combine_nodes, height_in, width_in, c_in, node.is_first, node.is_last)
        else:
            flops = get_flops(node, height_in, width_in, c_in, node.is_first, node.is_last)
        # runTime = flops / device.p / 1000 / 1000
        if algorithm_name == 'PSOGA':
            runTime = device.predict_time_by_node(node, flops, 0)
        else:
            runTime = device.predict_time_by_node(node, flops)
        # 遍历父节点
        for parent in node.parent_nodes:
            if not parent[0].visible:
                continue
            if parent[0].vmId != j:  # 只有不在一个设备执行才会有传输
                out_size = parent[1] * parent[2] * parent[3] * 4
                trans_time = out_size / B[devices[parent[0].vmId].type][devices[j].type] / 8 / 1024 / 1024 * 1000
                beginTime = parent[0].finishTime + trans_time
            else:
                beginTime = parent[0].finishTime
            node.beginTime = max(node.beginTime, beginTime)
        # 如果这个设备当前正有其他任务正在执行，需要等待，直到最后一个任务执行完成
        if len(device.period) > 0:
            node.beginTime = max(node.beginTime, device.period[len(device.period) - 1][1])
        node.finishTime = node.beginTime + runTime

        device.period.append([node.beginTime, node.finishTime])

        # df.loc[i] = [i, node, node.vmId, node.parent_nodes, node.child_nodes, node.k_size, node.k_num, node.stride, node.padding,
        #              flops, node.beginTime, node.finishTime]
    # writer = pd.ExcelWriter('output/node_time.xls')
    # df.to_excel(writer, startrow=0, startcol=0)
    # writer.save()
    totalTime = nodes[len(nodes) - 1].finishTime  # 总完成时间为最后一个节点的完成时间
    # totalTime = 1  # 总完成时间为最后一个节点的完成时间
    return totalTime
