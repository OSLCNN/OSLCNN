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
nowTime = 0
now = 0
leftTime = []
belong = 0
# cpu = 0
nowTimeCopy = 0
computeTimes = 0
transTimes = 0
waitTimes = 0
transFLOPS = 0
waitFLOPS = 0
transMaxEnd = 0
TimeLines = []

# 入侵代码
deviceTransIn = 0
deviceTransOut = 0
deviceFlops = 0

#优化版本
def get_fitness1(nodes_tmp, decision):
    # flag:是否按需分配
    global devices, B
    nodes = nodes_tmp
    # devices = devices_tmp
    # B = B_tmp
        #on_demand_assignment(decision)  # 按需分配卷积层
    return 1/on_demand_decision(decision,nodes)

def on_demand_decision(decision,node_tmp):
    #return 1
    global nodes, devices, B, leftTime, nowTime, nowTimeCopy
    # print(nowTime)
    nowTimeCopy = nowTime
    nodes = node_tmp
    time = 0
    decision = np.insert(decision, 0, belong)
    decisionL = []
    devI = 0
    i=0
    #print(decision)
    while i<nodes.__len__():
        begin_node = nodes[i]
        # print(i, nowTime)
        if begin_node.need_assign:  # 根据首结点来确定需要划分的节点的组合
            index = i
            begin_node_child_nodes = begin_node.child_nodes
            begin_node.vmId = decision[devI]
            devI+=1
            time_begin = timeNoConv(begin_node) + max(0,leftTime[begin_node.vmId]-nowTime)
            time = time + time_begin
            nextVmId = decision[devI]
            child_i = 0
            current_node = []
            nowTime = nowTime + time_begin
            while child_i < len(begin_node_child_nodes):
                child_node = begin_node_child_nodes[child_i][0]
                if not isinstance(child_node, ConvConcatNode):#孩子不是卷积层
                    index = index+1
                    child_i += 1
                    #计算时间
                    #time = time + timeTrans(begin_node,nextVmId)
                    i-=1
                    continue
                else:
                    #找到下一个水平分割的节点信息，存储一份就可以了
                    current_node.append(nodes[index+1])
                    #print("前后节点",current_node)
                    index = index+parameters.slice_num
                    child_i += parameters.slice_num
                    #计算时间，卷积层的
            deviceL = devices[begin_node.vmId]
            deviceR = devices[nextVmId]
                #print("卷积",timeConv(current_node,deviceL,deviceR,begin_node.vmId,nextVmId))
            time_conv = timeConv(current_node,deviceL,deviceR,begin_node.vmId,nextVmId, devices.__len__())
            time = time + time_conv
                #print(i,time)
            i+=len(begin_node_child_nodes)+1
            nowTime = nowTime + time_conv
        else:
            begin_node.vmId = decision[devI]
            devI+=1
            nextVmId = -1
            if devI<decision.__len__():
                nextVmId = decision[devI]
            time_no_conv = timeNoConvPlus(begin_node,nextVmId) + max(0,leftTime[begin_node.vmId]-nowTime)
            time = time + time_no_conv
            i = i+1
            nowTime = nowTime + time_no_conv
        #print(nowTime)
        #print(i,time)
    #print("time",time)
    #i=0
    # while i<nodes.__len__():
    #     begin_node = nodes[i]
    #     print(begin_node.vmId)
    #     i+=1
    nowTime = nowTimeCopy
    return time

def timeNoConv(node):
    if len(node.combine_nodes) > 0:
        combine_node = node.combine_nodes[0]
        flops = get_series_flops(node.combine_nodes, combine_node.height_in, combine_node.width_in, combine_node.c_in, combine_node.is_first, combine_node.is_last)
    else:
        flops = get_flops(node, node.height_in, node.width_in, node.c_in, node.is_first, node.is_last)
    runTime = devices[node.vmId].predict_time_by_node(node,flops)
    # print(node, flops)
    #print("只算自身计算时间", runTime,node.vmId)
    #print("只算自身计算时间", node.vmId)
    # if isinstance(node,ConvNode) == False:
    #     return node.flops/node.vmId
    return runTime

#卷积层节点，和连接卷积的前后节点
def timeConv(nodes,devL,devR,L,R,deviceNum, getTimeIndex = 0):
    global nowTime
    if nodes[0].LRTime[L][R]!=-1 and getTimeIndex == 0:
        return nodes[0].LRTime[L][R]
    speed = []
    height_outs = []
    for j in range(nodes.__len__()):
        height_outs.append(nodes[j].height_out)
    #print("卷积",node.vmId)
    #print("heights",height_outs)
    maxH = 0
    for i in range(nodes.__len__()):
        if height_outs[i]>height_outs[maxH]:
            maxH = i

    for i in range(devices.__len__()):
        temp = []
        temp.append(i)
        sumTime = 0
        for j in range(nodes.__len__()):
            nodes[j].vmId = i
            if j==maxH:
                sumTime += predict_time1(height_outs[j],devL,devR,nodes[j])
            else:
                sumTime += predict_time1(height_outs[j],devL,devR,nodes[j],-1)
            #sumTime += predict_time1(height_outs[j], devL, devR, nodes[j])
        temp.append(sumTime)
        speed.append(temp)
    #print("speed",speed)
    #nodes[0].vmId = 3
    #print("compare",predict_time1(4, devL, devR, nodes[0]),nodes[0].width_in, nodes[0].c_in, nodes[0].width_out, nodes[0].c_out ,predict_time(4, nodes[0].width_in, nodes[0].c_in, nodes[0].width_out, nodes[0].c_out ,devL, devR, nodes[0]))
    def takeSecond(ele):
        return ele[1]
    #排序
    speed.sort(key=takeSecond)
    #排序做好所有设备执行这个层的时间
    sum1 = 0
    sum2 = 0
    sum3 = 0
    #分配时间
    height_allocation = []
    time_allocation = []
    # print("speed",speed)
    #计算总时间
    #speed:[设备号,运行完整的层的时间]
    #height_allocation,time_allocation高度分配和时间分配
    leftTimeCopy = []
    # 用于标记是否参与分配
    isAllocate = []
    for i in range(speed.__len__()):
        isAllocate.append(0)
        sum1 += 1/speed[i][1]
        sum2 += speed[i][1]
        leftTimeCopy.append(max(0,leftTime[speed[i][0]]-nowTime)/speed[i][1])
        sum3 += leftTimeCopy[i]
    sum1Copy = sum1
    sum3Copy = sum3
    #print("sum1",sum1)
    noRunning = []
    for i in range(speed.__len__()):
        noRunning.append(False)
    while True:
        noZero = True
        for i in range(speed.__len__()):
            for j in range(nodes.__len__()):
                # print(height_outs[j], "/", speed[i][1], "/", sum1, "*", (1 + sum3), "-", height_outs[j], "*",
                #       leftTimeCopy[i], speed[i][1])
                height = max(0,
                             int(height_outs[j] / (speed[i][1] * sum1) * (1 + sum3) - height_outs[j] * leftTimeCopy[i]))
                # print(height_outs[j], "/", speed[i][1], "/", sum1, "*", (1 + sum3), "-", height_outs[j], "*",
                #       leftTimeCopy[i], height, speed[i][1], height)
                if height == 0 and noRunning[i] == False:
                    noRunning[i] = True
                    sum1Copy -= 1 / speed[i][1]
                    sum3Copy -= leftTimeCopy[i]
                    noZero = False
                if height == 0:
                    break
        if sum1Copy != 0:
            sum1 = sum1Copy
            sum3 = sum3Copy
        if noZero or sum1Copy == 0:
            break
    # print(noRunning)
    for i in range(speed.__len__()):
        height_allocation_outs = []
        sumTime = 0
        for j in range(nodes.__len__()):
            if noRunning[i] == True:
                height = 0
            else:
                height = max(0, int(height_outs[j] / (speed[i][1] * sum1) * (1 + sum3) - height_outs[j] * leftTimeCopy[i]))
                # print(height_outs[j],"/",speed[i][1],"/", sum1, "*", (1 + sum3) ,"-", height_outs[j] ,"*", leftTimeCopy[i], height)
            if isAllocate[speed[i][0]] == 0 and height > 0:
                isAllocate[speed[i][0]] = 1
                sumTime += max(0, leftTime[speed[i][0]] - nowTime)
            height_allocation_outs.append(height)
            nodes[j].vmId = speed[i][0]
            if j==maxH:
                sumTime += predict_time1(height,devL,devR,nodes[j])
            else:
                sumTime += predict_time1(height,devL,devR,nodes[j],-1)
        height_allocation.append(height_allocation_outs)
        time_allocation.append(sumTime)
    #print("time_all",time_allocation,nowTime, height_allocation)
    for j in range(nodes.__len__()):
        for i in range(deviceNum):
            height_outs[j]-=height_allocation[i][j]
    #print("before",time_allocation,height_outs,height_allocation)
    for i in range(nodes.__len__()):
        deviceAllocate = []
        for j in range(deviceNum):
            deviceAllocate.append(0)
        for j in range(height_outs[i]):
            #假定一个最小值
            minNow = 2*max(time_allocation)
            minIndex = 0
            for x in range(deviceNum):
                if noRunning[x] == True and sum1 != 0:
                    continue
                nodes[i].vmId = speed[x][0]
                tempx = time_allocation[x]
                time_allocation[x] += (predict_time1(height_allocation[x][i]+1+deviceAllocate[x], devL, devR, nodes[i])-predict_time1(height_allocation[x][i], devL, devR, nodes[i]))
                if isAllocate[speed[x][0]] == 0:
                    time_allocation[x] += max(0, leftTime[speed[x][0]] - nowTime)
                tempMax = max(time_allocation)
                if tempMax<minNow:
                    minIndex = x
                    minNow = tempMax
                time_allocation[x] = tempx
            # nodes[i].vmId = speed[minIndex][0]
            # time_allocation[minIndex] += (predict_time1(height_allocation[minIndex][i]+1, devL, devR, nodes[i], -1)-predict_time1(height_allocation[minIndex][i], devL, devR, nodes[i], -1))
            # height_allocation[minIndex][i] += 1
            deviceAllocate[minIndex] += 1
        for j in range(deviceNum):
            nodes[i].vmId = speed[j][0]
            time_allocation[j] += (predict_time1(height_allocation[j][i]+deviceAllocate[j], devL, devR, nodes[i])-predict_time1(height_allocation[j][i], devL, devR, nodes[i]))
            if isAllocate[speed[j][0]] == 0 and deviceAllocate[j] > 0:
                time_allocation[j] += max(0, leftTime[speed[j][0]] - nowTime)
                isAllocate[speed[j][0]] = 1
            height_allocation[j][i] += deviceAllocate[j]

    if nowTimeCopy >= max(leftTime):
        #print("now",nowTime)
        nodes[0].LRTime[L][R] = max(time_allocation)
    # else:
    #     print("time",time)
    #print("max",nodes[0].LRTime[L][R],time_allocation)
    # print(height_allocation,time_allocation,max(time_allocation),height_outs)

    if getTimeIndex == 1:
        global deviceTransIn, deviceTransOut, deviceFlops
        print("max", nodes[0].LRTime[L][R], time_allocation)
        for i in range(devices.__len__()):
            deviceTransIn = 0
            deviceTransOut = 0
            deviceFlops = 0
            for j in range(nodes.__len__()):
                nodes[j].vmId = speed[i][0]
                if j == maxH:
                    predict_time1(height_allocation[i][j], devL, devR, nodes[j], getTimeIndex = 1)
                else:
                    predict_time1(height_allocation[i][j], devL, devR, nodes[j], -1, 1)
            print(speed[i][0],"号设备", "传入时间",deviceTransIn,"传出时间",deviceTransIn,"flops",deviceFlops)
        maxWaitTime = 0
        for i in range(time_allocation.__len__()):
            maxWaitTime = max(maxWaitTime, time_allocation[i])
        for i in range(time_allocation.__len__()):
            global waitTimes, waitFLOPS
            if time_allocation[i] != 0:
                waitTimeTemp = maxWaitTime - time_allocation[i]
                waitTimes += waitTimeTemp
                waitFLOPS += devices[speed[i][0]].p * waitTimeTemp

    #print(speed)
    return max(time_allocation)

def timeNoConvPlus(node1, devR, getTimeIndex = 0):
    transTime = 0
    runTime = 0
    #print(node1,node2,devI)
    if devR!=-1:
        transTime = timeTrans(node1,devR,getTimeIndex)
    runTime += timeNoConv(node1)
    if getTimeIndex == 1:
        global computeTimes
        computeTimes += runTime
    #print("计算时间和发出去的时间",node1.vmId)
    #print("计算时间和发出去的时间",transTime+runTime)
    return transTime+runTime

def timeTrans(node1,devI,getTimeIndex = 0):
    transTime = 0.0
    if node1.vmId!=devI:
        out_size = node1.height_out*node1.width_out*node1.c_out*4
        transTime += out_size/B[devices[node1.vmId].type][devices[devI].type]*1000/8/1024/1024
    #print("只算传输时间",transTime)
    #print("只算传输时间")
    if getTimeIndex == 1:
        global transTimes, transFLOPS
        transTimes += transTime
        transFLOPS += transTime * (devices[devI].p + devices[node1.vmId].p)
    return transTime



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

######################################
def predict_time1(height_out, begin_device, end_device, between_node,trans = 1, getTimeIndex=0):
    if height_out == 0:
        return 0
    # is_first， is_last 表示这是否是第一个或最后一个切片，计算时需要增加一个padding，
    # 先根据输出大小，计算输入大小
    comp_time = 0
    between_device = devices[between_node.vmId]
    sum_time = 0
    width_in = between_node.width_in
    c_in = between_node.c_in
    width_out = between_node.width_out
    c_out = between_node.c_out
    combine_nodes = [between_node]
    if len(between_node.combine_nodes) > 0:
        combine_nodes = between_node.combine_nodes
    height_in = get_series_height_in(combine_nodes, height_out, between_node.is_first, between_node.is_last)
    flops = get_series_flops(combine_nodes, height_in, width_in, c_in, between_node.is_first, between_node.is_last)
    comp_time = flops / between_device.p / 1000 / 1000
    # comp_time += between_device.predict_time_by_type('conv', flops, cpu)
    begin_trans_time = get_transmission_time(height_in, width_in, c_in, begin_device, between_device)
    end_trans_time = get_transmission_time(height_out, width_out, c_out, between_device, end_device)
    #print(height_in,begin_trans_time,end_trans_time)
    #print(height_out,begin_trans_time,end_trans_time,comp_time)
    if getTimeIndex == 1:
        global deviceTransIn, deviceTransOut, deviceFlops
        global computeTimes, transTimes, transFLOPS, transMaxEnd
        computeTimes += comp_time
        transTimes += begin_trans_time + end_trans_time
        transFLOPS += between_device.p * (begin_trans_time + end_trans_time)

        deviceTransIn += begin_trans_time
        deviceTransOut += end_trans_time
        deviceFlops += flops

        transMaxTempEnd = end_device.p * end_trans_time
        transFLOPS += max(0, transMaxTempEnd - transMaxEnd)
        transMaxEnd = max(transMaxEnd, transMaxTempEnd)
    if trans==1:
        return begin_trans_time+end_trans_time+comp_time
    return end_trans_time+comp_time

# 根据输入大小 以及 node节点，计算节点的flops
def get_flops(node, height_in, width_in, c_in, is_first, is_last, height_out=1, width_out=1, c_out=1):
    height_out, width_out, c_out = get_output_shape(node, height_in, width_in, c_in, is_first, is_last)
    width_in = width_in + 2 * node.padding
    if is_first:
        height_in = height_in + node.padding
    if is_last:
        height_in = height_in + node.padding
    if isinstance(node, ConvNode) or isinstance(node, ConvConcatNode):
        flops = 2.0 * height_out * width_out * (c_in * node.k_size * node.k_size + 1) * node.k_num
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
        if height_in == width_in - 1:
            height_in = width_in
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

def getTimes(decision,node_tmp):
    #return 1
    global nodes, devices, B, leftTime, nowTime, nowTimeCopy, computeTimes, transTimes, waitTimes, transFLOPS, waitFLOPS
    computeTimes = 0
    transTimes = 0
    waitTimes = 0
    transFLOPS = 0
    waitFLOPS = 0
    # print(nowTime)
    nowTimeCopy = nowTime
    nodes = node_tmp
    time = 0
    decision = np.insert(decision, 0, belong)
    decisionL = []
    devI = 0
    i=0
    #print(decision)
    while i<nodes.__len__():
        begin_node = nodes[i]
        # print(i, nowTime)
        if begin_node.need_assign:  # 根据首结点来确定需要划分的节点的组合
            index = i
            begin_node_child_nodes = begin_node.child_nodes
            begin_node.vmId = decision[devI]
            devI+=1
            time_begin = timeNoConv(begin_node) + max(0,leftTime[begin_node.vmId]-nowTime)



            # 任务调度需要侵入的代码
            if len(begin_node.combine_nodes) > 0:
                combine_node = begin_node.combine_nodes[0]
                flops = get_series_flops(begin_node.combine_nodes, combine_node.height_in, combine_node.width_in,
                                         combine_node.c_in, combine_node.is_first, combine_node.is_last)
            else:
                flops = get_flops(begin_node, begin_node.height_in, begin_node.width_in, begin_node.c_in, begin_node.is_first, begin_node.is_last)
            print(i,"节点",decision[devI-1],"设备","下一个节点要划分，FLOPS",flops)



            # leftTime[begin_node.vmId]-nowTime > 0 说明有额外的设备等待时间
            waitTimes += 3 * max(0,leftTime[begin_node.vmId]-nowTime)
            for k in range(devices.__len__()):
                if k != begin_node.vmId:
                    waitFLOPS += max(0,leftTime[begin_node.vmId]-nowTime) * devices[k].p

            time = time + time_begin
            nextVmId = decision[devI]
            child_i = 0
            current_node = []
            nowTime = nowTime + time_begin
            while child_i < len(begin_node_child_nodes):
                child_node = begin_node_child_nodes[child_i][0]
                if not isinstance(child_node, ConvConcatNode):#孩子不是卷积层
                    index = index+1
                    child_i += 1
                    #计算时间
                    #time = time + timeTrans(begin_node,nextVmId)
                    i-=1
                    continue
                else:
                    #找到下一个水平分割的节点信息，存储一份就可以了
                    current_node.append(nodes[index+1])
                    #print("前后节点",current_node)
                    index = index+parameters.slice_num
                    child_i += parameters.slice_num
                    #计算时间，卷积层的
            deviceL = devices[begin_node.vmId]
            deviceR = devices[nextVmId]
                #print("卷积",timeConv(current_node,deviceL,deviceR,begin_node.vmId,nextVmId))
            global transMaxEnd
            transMaxEnd = 0

            time_conv = timeConv(current_node,deviceL,deviceR,begin_node.vmId,nextVmId, devices.__len__(), getTimeIndex = 1)
            time = time + time_conv
                #print(i,time)
            i+=len(begin_node_child_nodes)+1
            nowTime = nowTime + time_conv
        else:
            begin_node.vmId = decision[devI]
            devI+=1
            nextVmId = -1
            if devI<decision.__len__():
                nextVmId = decision[devI]
            timeNoConvPlusTemp = timeNoConvPlus(begin_node,nextVmId)


            # 任务调度需要侵入的代码
            if len(begin_node.combine_nodes) > 0:
                combine_node = begin_node.combine_nodes[0]
                flops = get_series_flops(begin_node.combine_nodes, combine_node.height_in, combine_node.width_in,
                                         combine_node.c_in, combine_node.is_first, combine_node.is_last)
            else:
                flops = get_flops(begin_node, begin_node.height_in, begin_node.width_in, begin_node.c_in, begin_node.is_first, begin_node.is_last)
            transTime1 = 0
            if nextVmId != -1:
                transTime1 = timeTrans(begin_node, nextVmId)
            print(i,"节点",decision[devI-1],"设备","下一个设备是",nextVmId,"FLOPS",flops,"transTime",transTime1)
            time_no_conv = 3 * timeNoConvPlusTemp + max(0,leftTime[begin_node.vmId]-nowTime)
            # leftTime[begin_node.vmId]-nowTime > 0 说明有额外的设备等待时间
            waitTimes += max(0,leftTime[begin_node.vmId]-nowTime)
            for k in range(devices.__len__()):
                if k != nextVmId:
                    waitFLOPS += timeNoConvPlusTemp * devices[k].p
            time = time + time_no_conv
            i = i+1
            nowTime = nowTime + time_no_conv
        #print(nowTime)
        #print(i,time)
    #print("time",time)
    #i=0
    # while i<nodes.__len__():
    #     begin_node = nodes[i]
    #     print(begin_node.vmId)
    #     i+=1
    nowTime = nowTimeCopy
    print("传输和等待时间", transTimes, waitTimes, transFLOPS, waitFLOPS)
    return time

if __name__ == '__main__':
    print(3)