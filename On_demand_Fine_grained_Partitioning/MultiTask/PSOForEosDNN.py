#一次PSO对应所有模型

import copy
import random

from On_demand_Fine_grained_Partitioning.Node import CombineNode, ConvNode, FcNode, PoolNode, ConvConcatNode, ConcatNode, Node
from On_demand_Fine_grained_Partitioning import parameters
from On_demand_Fine_grained_Partitioning.Util import get_fitness1
from On_demand_Fine_grained_Partitioning.MultiTask.MyModel import MyModel
from On_demand_Fine_grained_Partitioning.MultiTask import MyUtil

fitness_list = []      # 每个种群的最优值
global_fitness_list = []   # 全局最优值，肯定是非递增的,列表最后那个元素就是全局最优的QoE
# global_decision = []
global_best_fitness = float('inf')
global_best_decision = []


class PSOPartition:
    def __init__(
        self,
        models: list,
        devices: list,
        B,
        max_pop_size=parameters.max_pop_size,
        max_iter_size=parameters.max_iter_size    # 最大迭代次数
    ):
        # 初始化
        global fitness_list, global_fitness_list, global_best_fitness, global_best_decision
        fitness_list = []
        global_fitness_list = []
        global_best_fitness = float('inf')
        global_best_decision = []
        self.models = models
        self.devices = devices
        self.deviceNum = 0
        self.edgeNum = 0
        self.cloudNum = 0
        for device in devices:
            if device.type == 0:
                self.deviceNum = self.deviceNum + 1
            elif device.type == 1:
                self.edgeNum = self.edgeNum + 1
            else:
                self.cloudNum = self.cloudNum + 1
        self.B = B
        self.max_pop_size = max_pop_size
        self.max_iter_size = max_iter_size
        self.psoLength = 0
        self.nodes_len = []
        for j in range(models.__len__()):
            nodes = self.models[j].nodes
            sum = nodes.__len__()
            print(nodes.__len__())
            for i in range(nodes.__len__()):
                if nodes[i].child_nodes.__len__() > 1:
                    node = nodes[i]
                    for k in range(node.child_nodes.__len__()):
                        child_node = node.child_nodes[k][0]
                        if child_node.parent_nodes.__len__() == 1:
                            sum -= 1
                    sum += 1
            self.nodes_len.append(sum)
            self.psoLength += sum
        print(self.psoLength)
        self.transTime = 0
        self.waitTime = 0
        self.transFlops = 0
        self.waitFlops = 0

    def run(self):
        global global_best_fitness, global_best_decision
        initPop = self.init_pop()        # 均匀的分布初始化种群
        initVec = self.init_vec_randomly()    # 随机的初始化种群

        # 保存 初始种群的 QoE
        current_best_decision, current_best_fitness = self.find_best(initPop)
        #print("best",current_best_decision,current_best_fitness)
        fitness_list.append(current_best_fitness)
        global_fitness_list.append(current_best_fitness)

        global_best_fitness = current_best_fitness
        global_best_decision = copy.copy(current_best_decision)

        self.run_pso(initPop, initVec)
        self.getfitness(global_best_decision, 1)
        print("传输时间和等待时间", self.transTime, self.waitTime, self.transFlops, self.waitFlops)

        return global_best_decision, fitness_list, global_fitness_list

    def find_best(self, pops):
        best_fitness = float('inf')
        best_index = 0
        for i in range(len(pops)):
            fitness = self.getfitness(pops[i])
            #print(i,"个种群的结果:",fitness)
            if fitness < best_fitness:
                best_fitness = fitness
                best_index = i
        return copy.copy(pops[best_index]), best_fitness

    def run_pso(self, pop, vec):
        global global_best_fitness, global_best_decision
        pBestFitness = []
        pBestList = []

        for i in range(len(pop)):
            fitness = self.getfitness(pop[i])
            pBestList.append(pop[i])       # 设置pBest为均匀初始化的种群
            pBestFitness.append(fitness)
            if global_best_fitness > fitness:  # 选最小适应度值
                global_best_fitness = fitness
                global_best_decision = copy.copy(pop[i])

        record_best_fitness = global_best_fitness
        record_best_time = 1

        for i in range(self.max_iter_size):   # 迭代更新速度和位置
            # print("迭代次数"+i.__str__())
            pop_temp = []
            vec_temp = []
            w = parameters.w_max-(parameters.w_max-parameters.w_min)*i/parameters.max_iter_size
            for j in range(len(vec)):
                vec1 = vec[j]      # 从初始的随机种群，选择第j个粒子
            # v(t) = w * v(t - 1) + c1 * r1 * (gbest(t - 1)) + c2 * r2 * (pbest(t - 1))  速度更新公式
                for k in range(len(vec1)):      # 计算速度
                    vec1[k] = int(w * vec1[k] + random.random() * 2 * (abs(global_best_decision[k] - pop[j][k])) +
                                  random.random() * 2 * (abs(pBestList[j][k] - pop[j][k]))) % len(self.devices)
                vec_temp.append(vec1)
            for j in range(len(pop)):
                pop1 = pop[j]
                vec2 = vec_temp[j]
                for k in range(len(pop1)):
                    pop1[k] = int(pop1[k] + vec2[k]) % len(self.devices)        # 计算新的位置
                pop_temp.append(pop1)

            for j in range(len(pop_temp)):
                pop[j] = pop_temp[j]    # 更新位置
                vec[j] = vec_temp[j]     # 更新速度

            best_fitness = float('inf')
            pBest = 0
            for j in range(len(pop)):
                fitness = self.getfitness(pop[j])
                if pBestFitness[j] > fitness:
                    pBestList[j] = copy.copy(pop[j])
                    pBestFitness[j] = fitness
                if global_best_fitness > fitness:      # 更新全局最优解gBest
                    global_best_fitness = fitness
                    global_best_decision = copy.copy(pop[j])
                if fitness < best_fitness:
                    best_fitness = fitness
                    pBest = i
            fitness_list.append(pBest)

            print(i,global_best_fitness,global_best_decision)
            if record_best_fitness - global_best_fitness > 0.0001:
                record_best_fitness = global_best_fitness
                record_best_time = 1
            else:
                record_best_time += 1
                if record_best_time == 2000:
                    break
            fitness_list.append(pBest)
            global_fitness_list.append(global_best_fitness)
        return global_best_decision

    def init_vec_randomly(self):     # 随机初始化种群
        population = []
        #print(len(self.nodes))
        for i in range(self.max_pop_size):
            p = []
            for j in range(self.psoLength):
                p.append(random.randint(0, self.deviceNum + self.edgeNum + self.cloudNum - 1))
            population.append(p[:])
        return population

    def init_pop(self):      # 均匀初始化种群
        num = self.deviceNum + self.edgeNum + self.cloudNum
        if num < 3:
            return self.init_vec_randomly()
        population = []
        device_num = num // 3
        edge_num = (num - device_num) // 2
        cloud_num = num - device_num - edge_num
        for i in range(self.max_pop_size):
            if i < self.max_pop_size / 3:    # 1/3种群选择边缘端或本地
                p = []
                for j in range(self.psoLength):
                    p.append(random.randint(0, device_num + edge_num - 1))
                population.append(p[:])
            elif i < self.max_pop_size * 2 / 3:   # 1/3种群选择边缘端或云端
                p = []
                for j in range(self.psoLength):
                    p.append(random.randint(device_num, device_num + edge_num + cloud_num - 1))
                population.append(p[:])
            else:   # 1/3种群选择本地或云端
                p = []
                for j in range(self.psoLength):
                    deviceIndex = random.randint(0, device_num + cloud_num - 1)
                    if deviceIndex < device_num:        # 在本地
                        p.append(deviceIndex)
                    else:
                        p.append(deviceIndex + edge_num)       # 在云端
                population.append(p[:])
        return population

    def getfitness(self, pop:list, getTimeIndex = 0):
        deviceTimes = []
        for i in range(self.devices.__len__()):
            deviceTimes.append(0)
        number = 0
        time = 0
        for i in range(self.models.__len__()):
            nodes: Node = self.models[i].nodes
            nodesLen = self.nodes_len[i]
            decision = pop[number: number + nodesLen]
            decision.insert(0, self.models[i].device)
            number += nodesLen
            time = self.getTime(nodes, decision, deviceTimes, 0, getTimeIndex)
        return max(deviceTimes)



    def getTime(self, nodes, decision, deviceTime, time, getTimeIndex = 0):
        decI = 0
        i = 0
        #print(decision)
        while i < nodes.__len__():
            #print(i, decI)
            begin_node = nodes[i]
            nodes[i].vmId = decision[decI]
            decI += 1
            i += 1
            if begin_node.child_nodes.__len__() > 1:
                child_len = 0
                for j in range(begin_node.child_nodes.__len__()):
                    child_node = begin_node.child_nodes[j][0]
                    if child_node.parent_nodes.__len__() == 1:
                        child_node.vmId = decision[decI]
                        child_len += 1
                decI += 1
                i += child_len
        i = 0
        # print(nodes.__len__(), decision)
        while i < nodes.__len__():
            begin_node = nodes[i]
            if begin_node.child_nodes.__len__() > 1:  # 根据首结点来确定需要划分的节点的组合
                maxTime = 0
                begin_node_child_nodes = begin_node.child_nodes
                # 更新计算时间
                timeTmp = max(time, deviceTime[begin_node.vmId]) + MyUtil.timeNoConv(begin_node)
                if getTimeIndex == 1:
                    self.waitTime += max(0, time - deviceTime[begin_node.vmId])
                    self.waitFlops += max(0, time - deviceTime[begin_node.vmId]) * self.devices[begin_node.vmId].p
                deviceTime[begin_node.vmId] = timeTmp
                time = timeTmp
                child_i = 0
                while child_i < len(begin_node_child_nodes):
                    child_node = begin_node_child_nodes[child_i][0]
                    if not isinstance(child_node, ConvConcatNode):  # 孩子不是卷积层
                        child_i += 1
                        nextVmId = child_node.child_nodes[0][0].vmId
                        # 计算时间
                        maxTime = max(maxTime, deviceTime[begin_node.vmId] + MyUtil.timeTrans(begin_node,nextVmId))
                        if getTimeIndex == 1:
                            self.transTime += MyUtil.timeTrans(begin_node,nextVmId)
                            self.transFlops += MyUtil.timeTrans(begin_node, nextVmId) * self.devices[nextVmId].p
                        i -= 1
                        continue
                    else:
                        # 找到下一个水平分割的节点信息，存储一份就可以了
                        # print("前后节点",current_node)
                        timeTmp = max(time, deviceTime[child_node.vmId]) + MyUtil.timeTrans(begin_node, child_node.vmId) + MyUtil.timeNoConv(child_node)
                        if getTimeIndex == 1:
                            self.transTime += MyUtil.timeTrans(begin_node,child_node.vmId)
                            self.waitTime += max(0, time - deviceTime[child_node.vmId])
                            self.transFlops += MyUtil.timeTrans(begin_node, child_node.vmId) * self.devices[
                                child_node.vmId].p
                            self.waitFlops += max(0, time - deviceTime[child_node.vmId]) * self.devices[
                                child_node.vmId].p
                        deviceTime[child_node.vmId] = timeTmp
                        nextVmId = child_node.child_nodes[0].vmId
                        maxTime = max(maxTime, timeTmp + MyUtil.timeTrans(child_node, nextVmId))

                        child_i += 1
                        # 计算时间，卷积层的
                i += len(begin_node_child_nodes) + 1
                time = maxTime
            else:
                nextVmId = -1
                if i + 1 < decision.__len__():
                    nextVmId = decision[i + 1]
                # 考虑到设备的传输时间不占用设备的计算时间，因此deviceTime里面只能存放计算时间
                timeTmp = max(time, deviceTime[begin_node.vmId]) + MyUtil.timeNoConv(begin_node)
                # print(timeTmp)
                if getTimeIndex == 1:
                    self.waitTime += max(0,  time - deviceTime[begin_node.vmId])
                    self.waitFlops += max(0, time - deviceTime[begin_node.vmId]) * self.devices[begin_node.vmId].p
                deviceTime[begin_node.vmId] = timeTmp
                if nextVmId != -1:
                    time = timeTmp + MyUtil.timeTrans(begin_node, nextVmId)
                    if getTimeIndex == 1:
                        self.transTime += MyUtil.timeTrans(begin_node, nextVmId)
                        self.transFlops += MyUtil.timeTrans(begin_node, nextVmId) * self.devices[nextVmId].p
                else:
                    time = timeTmp
                i = i + 1
        return time



