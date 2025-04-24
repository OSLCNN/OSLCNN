#EosDNN: An Efficient Offloading Scheme for DNN Inference Acceleration in Local-Edge-Cloud Collaborative Environments
import copy
import time
import random
import matplotlib.pyplot as plt
import numpy as np

from On_demand_Fine_grained_Partitioning import Util, readJson, ConstructEnvironment, just_hori_partition
from On_demand_Fine_grained_Partitioning.Node import ConcatNode, ConvNode, ConvConcatNode, CombineNode, FcNode
from On_demand_Fine_grained_Partitioning.MultiTask.PSOForEosDNN import PSOPartition
from On_demand_Fine_grained_Partitioning import parameters
from On_demand_Fine_grained_Partitioning.MultiTask.Demo1 import exportModels
from On_demand_Fine_grained_Partitioning.MultiTask import PSONoPartition, MyModel, MyParameters, TimeLineUtil, testMyPartition, MyUtil


plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）

devices = []
B = []
nodes = []
model_name = []
# cpu = 0
# slice_num = parameters.slice_num      # 水平切割的份数

import sys  # 导入sys模块
sys.setrecursionlimit(3000)  # 将默认的递归深度修改为3000
plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）


# def plot(list, title):
#
#     plt.plot(np.arange(len(list)), list)
#     plt.ylabel('目标值')
#     plt.xlabel('迭代次数')
#     plt.title(title)
    # plt.show()


class ModelPro:
    def __init__(self):
        self.GoogleNet = "GoogleNet"
        self.AlexNet = "AlexNet"
        self.ResNet50 = "ResNet50"
        self.Vgg16 = "Vgg16"
        self.TestNet = "TestNet"
        self.YOLO = "YOLO"


class AlgorithmPro:
    def __init__(self):
        self.GA = "GA"
        self.PSO = "PSO"

def horizontal_partition(nodes_tmp):    # 删除尾结点

    if parameters.slice_num == 0:   # 不切割
        return nodes_tmp[0]

    # 合并conv层和分支上的所有节点
    nodes_tmp = readJson.combine_branch(readJson.combine_conv(nodes_tmp))
    id = 999
    length = len(nodes_tmp)
    for i in range(length):
        node = nodes_tmp[i]
        # 是否需要水平切割
        need_partition = isinstance(node, ConvNode) or (isinstance(node, CombineNode) and node.hasConv)
        if need_partition:

            parent = node.parent_nodes[0]       # 默认conv层的父节点,子节点个数都为1个
            child = node.child_nodes[0]

            slice_nodes = []
            for j in range(parameters.slice_num):
                if isinstance(node, ConvNode):
                    conv_concat_node = ConvConcatNode(id, 'conv_concat', 'node', node.k_size, node.k_num, node.stride, node.padding)
                else:   # 合并后的CombineNode，没有k_size等属性
                    conv_concat_node = ConvConcatNode(id, 'conv_concat', 'node', 0, 0, 0, 0)
                if parameters.slice_num > 1 and j != parameters.slice_num-1:
                    conv_concat_node.is_last = False
                if parameters.slice_num > 1 and j != 0:
                    conv_concat_node.is_first = False
                conv_concat_node.hasConv = True
                slice_nodes.append(conv_concat_node)

            # 切割前：parent -- node -- child
            # 切割后：       /   node1  \
            #        parent ——  node2 -- child
            #               \   node3  /

            # 修改 child到 node1,2,3的连接
            child_length_sum = child[1]
            child_length_arr = []
            for index, node_list in enumerate(child[0].parent_nodes):
                if node_list[0] == node:
                    break

            slice_num_index = parameters.slice_num
            for conv_concat_node in reversed(slice_nodes):
                child_length = child_length_sum // slice_num_index
                child_length_arr.append(child_length)
                slice_num_index -= 1
                child_length_sum -= child_length
                child[0].parent_nodes.insert(index, [conv_concat_node, child_length, child[2], child[3]])
            child[0].parent_nodes.remove(node_list)

            for index, node_list in enumerate(parent[0].child_nodes):
                if node_list[0] == node:
                    break
            for index_tmp, conv_concat_node in enumerate(reversed(slice_nodes)):    # 倒序遍历
                combine_nodes = []
                if len(node.combine_nodes) == 0:
                    combine_nodes = [node]
                else:
                    combine_nodes = node.combine_nodes
                parent_length = Util.get_series_height_in(combine_nodes, child_length_arr[index_tmp], index_tmp == 0, index_tmp == parameters.slice_num-1)
                # 修改 parent到node1,2,3的连接
                parent[0].child_nodes.insert(index, [conv_concat_node, parent_length, parent[2], parent[3]])
                # 修改 node1,2,3到parent的连接
                conv_concat_node.parent_nodes = [[parent[0], parent_length, parent[2], parent[3]]]
                # 修改 node1,2,3到child的连接
                conv_concat_node.child_nodes = [[child[0], child_length_arr[index_tmp], child[2], child[3]]]
            parent[0].child_nodes.remove(node_list)
            parent[0].need_assign = True

            # 设置node1,2,3的combine_nodes，height_in,width_in,c_in等参数都和原来一样
            for conv_concat_node in slice_nodes:
                conv_concat_node.combine_nodes = node.combine_nodes
                # conv_concat_node.is_partitioned = True
                conv_concat_node.height_in = parent[1]
                conv_concat_node.width_in = parent[2]
                conv_concat_node.c_in = parent[3]
                conv_concat_node.height_out = child[1]
                conv_concat_node.width_out = child[2]
                conv_concat_node.c_out = child[3]
    return nodes_tmp[0]


time_list = []
decision_list = []


# 不分割时，是串行执行的，在指定设备device上执行
def no_partition(device):
    comp_time = 0
    trans_time = 0
    ss = 0
    if device.id != 0:
        trans_time = nodes[0].height_out * nodes[0].width_out * nodes[0].c_out * 4 / B[0][device.type] / 8 / 1024 / 1024 * 1000
    for i in range(1, len(nodes)):
        combine_nodes = [nodes[i]]
        if len(nodes[i].combine_nodes) > 0:
            combine_nodes = nodes[i].combine_nodes
        for node in combine_nodes:
            # tmp = node.flops / device.p / 1000 / 1000
            # print(node.flops)
            ss +=1
            tmp = device.predict_time_by_node(node, node.flops)
            comp_time = comp_time + tmp
    return comp_time + trans_time


def put_all_device_cloud():
    # 全在本地
    device_time = no_partition(devices[0])
    time_list.append(device_time)
    # decision_list.append([])
    print("全在本地执行:" + str(device_time))

    # 全在云
    cloud_time = no_partition(devices[len(devices) - 1])
    # time_list.append(cloud_time)
    print("全在云端执行:" + str(cloud_time))
    cloud_time = 0

    # 全在边缘
    edge_min_time = float('inf')
    edge_time = 0
    edge_decision = []
    for i in range(1, len(devices)):
        edge_decision = [devices[i].id]
        edge_time = no_partition(devices[i])
        if edge_time < edge_min_time:
            edge_min_time = edge_time
    if edge_time == 0:  # 没有其他设备，只有这一个
        edge_time = device_time
    time_list.append(edge_time)
    decision_list.append(edge_decision)
    print("全在边缘执行:" + str(edge_time))
    return device_time, edge_time, cloud_time


def multi_partition(nodes_tmp):   # flag : True 按需划分，False 不按需划分， picture_pos：图的位置
    start_time = time.time()
    global_decision = [0]
    pso = PSOPartition(nodes_tmp, devices, B)
    global_decision, fitness_list, global_fitness_list = pso.run()
    print(global_decision)
    global_time = global_fitness_list[len(global_fitness_list) - 1]
    end_time = time.time()

    # print("时间：" + str(global_time) + "节点数量： " + str(len(nodes_tmp)) + ", 算法运行时间： " + str(end_time-start_time) + "s")
    time_list.append(global_time)
    decision_list.append(global_decision)
    return global_time


def multi_hori_partition(flag, picture_pos):
    nodes_tmp = copy.deepcopy(nodes)
    # begin_node = horizontal_partition_needs(nodes_tmp)
    begin_node = horizontal_partition(nodes_tmp)
    sort_nodes = readJson.get_sort_nodes(begin_node)
    multi_hori_time = multi_partition(flag, sort_nodes, picture_pos)
    return multi_hori_time

def models_Vertical_Cut(models):
    for i in range(models.__len__()):
        model: MyModel.MyModel = models[i]
        model.nodes = readJson.combine_norm_relu(model.nodes)
        model.nodes = readJson.get_sort_nodes(readJson.combine_conv(model.nodes))
        model.nodes = readJson.combine_branch(model.nodes[0])
        model.nodes = readJson.get_sort_nodes(model.nodes[0])


def run_fix_bp(devices_p, B_p, nodes_p, leftTime, nowTime):

    global model_name, devices, B, nodes, time_list, decision_list
    MyUtil.devices = devices = devices_p
    MyUtil.B = B = B_p
    MyUtil.leftTime = leftTime
    MyUtil.belong = nodes_p.device
    MyUtil.nowTime = nowTime
    nodes = nodes_p.nodes
    time_list.clear()
    decision_list.clear()

    # 1,2,3 全在本地执行，全在云执行，全在某个边缘执行
    # put_all_device_cloud()

    # 4. 多粒度切割
    # multi_nodes = copy.deepcopy(nodes)
    # multi_nodes = readJson.combine_branch(multi_nodes[0])  # 合并分支
    #multi_partition('PSOGA', multi_nodes, 131)    # 一行三列的第一张图
    # time_list.append(1)
    # decision_list.append([0])

    # 5. 水平切割
    #hori_partition()
    # time_list.append(1)
    # decision_list.append([0])

    nodes = readJson.combine_branch(nodes[0])  # 合并分支
    # 6. 多粒度+平均分水平切割
    #multi_hori_partition('FP', 132)

    # 7. 多粒度+水平切割
    multi_hori_partition('OFP', 133)
    # time_list.append(1)
    # decision_list.append([0])

    #name_list = ['Local', 'Central', 'PSOGA', 'EdgeLD', 'FPM', 'OFPM']
    name_list = ['OFPM']
    print(name_list)
    print('时间' + str(time_list))
    print('决策结果' + str(decision_list))

    # plt.show()
    return time_list

def test(num):
    parameters.slice_num = 0
    modelNum = num
    modelPools = [
        [0, "AlexNet", 227, 3],
        [0, "GoogleNet", 227, 3],
        [0, "Vgg16", 224, 3],
        [0, "ResNet50", 224, 3],
        [0, "YOLO", 448, 3]
    ]
    models = []
    for i in range(modelNum):
        poolNumber = modelPools[4]
        # poolNumber = modelPools[i % 5]
        models.append(MyModel.MyModel(poolNumber[0], poolNumber[1], poolNumber[2], poolNumber[3]))
    models_Vertical_Cut(models)
    # 1. 固定环境，可以 单独 运行各个模型，分别比较 收敛程度+++，时延
    devices, B = ConstructEnvironment.construct_environment_real5() # 指定环境
    MyUtil.devices = devices
    MyUtil.B = B
    pso = PSOPartition(models, devices, B)
    ans1, ans2, ans3 = pso.run()
    p = time.time()
    sum = 0
    #test_partition(nodes,'133')
    print("运行时间",time.time()-p)
    return num, min(ans3)

if __name__ == '__main__':
    ansList = []
    now = time.time()
    print(test(5))
    print(time.time() - now)
    # for i in range(1,9):
    #     ansList.append(test(i))
    # for i in range(ansList.__len__()):
    #     print(ansList[i])
