import copy
import random

from On_demand_Fine_grained_Partitioning.Node import CombineNode, ConvNode, FcNode, PoolNode, ConvConcatNode, ConcatNode
from On_demand_Fine_grained_Partitioning import parameters
from On_demand_Fine_grained_Partitioning.Util import get_fitness1, getTimes

fitness_list = []      # 每个种群的最优值
global_fitness_list = []   # 全局最优值，肯定是非递增的,列表最后那个元素就是全局最优的QoE
# global_decision = []
global_best_fitness = float('inf')
global_best_decision = []


class PSO:
    def __init__(
        self,
        algorithm,
        nodes,
        devices,
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
        self.algorithm = algorithm
        self.nodes = nodes
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
        self.node1 = 0
        i=0
        while i<self.nodes.__len__():
            print(self.nodes[i], self.nodes[i].need_assign)
            if not isinstance(self.nodes[i] ,ConvConcatNode):
                self.node1+=1
            else:
                for k in range(devices.__len__()):
                    timeI = []
                    for j in range(devices.__len__()):
                        timeI.append(-1)
                    self.nodes[i].LRTime.append(timeI)
            i+=1
        print(self.node1)


    def run(self):
        global global_best_fitness, global_best_decision
        initPop = self.init_pop()        # 均匀的分布初始化种群
        initVec = self.init_vec_randomly()    # 随机的初始化种群

        # 保存 初始种群的 QoE
        current_best_decision, current_best_fitness = self.find_best(initPop)
        fitness_list.append(current_best_fitness)
        global_fitness_list.append(current_best_fitness)

        global_best_fitness = current_best_fitness
        global_best_decision = copy.copy(current_best_decision)

        self.run_pso(initPop, initVec)
        if self.algorithm == 'PSOGA':
            global_fitness_list[-1] = 1 / get_fitness1('',self.nodes, global_best_decision)
        getTimes(global_best_decision,self.nodes)
        return global_best_decision, fitness_list, global_fitness_list

    def run_adv(self):
        global global_best_fitness, global_best_decision

        initPop = self.init_pop()  # 均匀的分布初始化种群
        initVec = self.init_vec_randomly()  # 随机的初始化种群

        # 保存 初始种群的 QoE
        current_best_decision, current_best_fitness = self.find_best(initPop)
        fitness_list.append(current_best_fitness)
        global_fitness_list.append(current_best_fitness)

        global_best_fitness = current_best_fitness
        global_best_decision = copy.copy(current_best_decision)

        self.run_pso(initPop, initVec)
        if self.algorithm == 'PSOGA':
            global_fitness_list[-1] = 1 / get_fitness1('',self.nodes, global_best_decision)
        return global_best_decision, fitness_list, global_fitness_list

    def find_best(self, pops):
        best_fitness = float('inf')
        best_index = 0
        for i in range(len(pops)):
            fitness = 1 / get_fitness1(self.algorithm,self.nodes,  pops[i])
            if fitness < best_fitness:
                best_fitness = fitness
                best_index = i
        return copy.copy(pops[best_index]), best_fitness

    def run_pso(self, pop, vec):
        global global_best_fitness, global_best_decision
        pBestFitness = []
        pBestList = []

        for i in range(len(pop)):
            fitness = 1 / get_fitness1(self.algorithm,self.nodes,  pop[i])
            pBestList.append(pop[i])       # 设置pBest为均匀初始化的种群
            pBestFitness.append(fitness)
            if global_best_fitness > fitness:  # 选最小适应度值
                global_best_fitness = fitness
                global_best_decision = copy.copy(pop[i])

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
                fitness = 1 / get_fitness1(self.algorithm,self.nodes,  pop[j])
                if pBestFitness[j] > fitness:
                    pBestList[j] = copy.copy(pop[j])
                    pBestFitness[j] = fitness
                if global_best_fitness > fitness:      # 更新全局最优解gBest
                    global_best_fitness = fitness
                    global_best_decision = copy.copy(pop[j])
                if fitness < best_fitness:
                    best_fitness = fitness
                    pBest = i

            print(i,global_best_fitness,global_best_decision)
            fitness_list.append(pBest)
            global_fitness_list.append(global_best_fitness)
        # return global_best_decision

    def init_vec_randomly(self):     # 随机初始化种群
        population = []
        #print(len(self.nodes))
        for i in range(self.max_pop_size):
            p = []
            for j in range(self.node1-1):
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
                for j in range(self.node1-1):
                    p.append(random.randint(0, device_num + edge_num - 1))
                population.append(p[:])
            elif i < self.max_pop_size * 2 / 3:   # 1/3种群选择边缘端或云端
                p = []
                for j in range(self.node1-1):
                    p.append(random.randint(device_num, device_num + edge_num + cloud_num - 1))
                population.append(p[:])
            else:   # 1/3种群选择本地或云端
                p = []
                for j in range(self.node1-1):
                    deviceIndex = random.randint(0, device_num + cloud_num - 1)
                    if deviceIndex < device_num:        # 在本地
                        p.append(deviceIndex)
                    else:
                        p.append(deviceIndex + edge_num)       # 在云端
                population.append(p[:])
        return population

    # 初始版，备份
    def init_pop_backup(self):  # 均匀初始化种群
        population = []
        for i in range(self.max_pop_size):
            if i < self.max_pop_size / 3:  # 1/3种群选择边缘端或本地
                p = []
                for j in range(len(self.nodes) - 1):
                    p.append(random.randint(0, self.deviceNum + self.edgeNum - 1))
                population.append(p[:])
            elif i < self.max_pop_size * 2 / 3:  # 1/3种群选择边缘端或云端
                p = []
                for j in range(len(self.nodes) - 1):
                    p.append(random.randint(self.deviceNum, self.deviceNum + self.edgeNum + self.cloudNum - 1))
                population.append(p[:])
            else:  # 1/3种群选择本地或云端
                p = []
                for j in range(len(self.nodes) - 1):
                    deviceIndex = random.randint(0, self.deviceNum + self.cloudNum - 1)
                    if deviceIndex < self.deviceNum:  # 在本地
                        p.append(deviceIndex)
                    else:
                        p.append(deviceIndex + self.edgeNum)  # 在云端
                population.append(p[:])
        return population


