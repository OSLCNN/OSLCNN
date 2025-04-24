import copy
import random

from On_demand_Fine_grained_Partitioning.Node import CombineNode, ConvNode, FcNode, PoolNode, ConvConcatNode, ConcatNode
from On_demand_Fine_grained_Partitioning import parameters
from On_demand_Fine_grained_Partitioning.Util import get_fitness1
from On_demand_Fine_grained_Partitioning.MultiTask.MyModel import MyModel

fitness_list = []      # 每个种群的最优值
global_fitness_list = []   # 全局最优值，肯定是非递增的,列表最后那个元素就是全局最优的QoE
# global_decision = []
global_best_fitness = float('inf')
global_best_decision = []


class PSONoPartition:
    def __init__(
        self,
        algorithm,
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
        self.algorithm = algorithm
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
        computeTimes, transTimes, resultTimes, transFLOPS = self.getDetailedTime(global_best_decision)
        print("计算时间和传输时间，传输所消耗的FLOPS", computeTimes,transTimes, resultTimes, transFLOPS)
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
            fitness_list.append(pBest)
            global_fitness_list.append(global_best_fitness)
        return global_best_decision

    def init_vec_randomly(self):     # 随机初始化种群
        population = []
        #print(len(self.nodes))
        for i in range(self.max_pop_size):
            p = []
            for j in range(self.models.__len__()):
                p.append(random.randint(0, self.deviceNum + self.edgeNum + self.cloudNum - 1))
            population.append(p[:])
        return population

    def init_pop(self):      # 均匀初始化种群
        num = self.deviceNum + self.edgeNum + self.cloudNum
        modelNum = self.models.__len__()
        if num < 3:
            return self.init_vec_randomly()
        population = []
        device_num = num // 3
        edge_num = (num - device_num) // 2
        cloud_num = num - device_num - edge_num
        for i in range(self.max_pop_size):
            if i < self.max_pop_size / 3:    # 1/3种群选择边缘端或本地
                p = []
                for j in range(modelNum):
                    p.append(random.randint(0, device_num + edge_num - 1))
                population.append(p[:])
            elif i < self.max_pop_size * 2 / 3:   # 1/3种群选择边缘端或云端
                p = []
                for j in range(modelNum):
                    p.append(random.randint(device_num, device_num + edge_num + cloud_num - 1))
                population.append(p[:])
            else:   # 1/3种群选择本地或云端
                p = []
                for j in range(modelNum):
                    deviceIndex = random.randint(0, device_num + cloud_num - 1)
                    if deviceIndex < device_num:        # 在本地
                        p.append(deviceIndex)
                    else:
                        p.append(deviceIndex + edge_num)       # 在云端
                population.append(p[:])
        return population

    def getfitness(self, pop:list):
        deviceTimes = []
        for i in range(self.devices.__len__()):
            deviceTimes.append(0)
        for i in range(pop.__len__()):
            modelTime = self.getTime(pop[i],i)
            # print(pop[i],modelTime)
            deviceTimes[pop[i]] += modelTime
        return max(deviceTimes)

    def getTime(self, device, index):
        model: MyModel = self.models[index]
        totalTime = 0
        #print("index",index)
        # print(model.FLOPS)
        for i in range(model.nodes.__len__()):
            totalTime += model.nodes[i].flops / self.devices[device].p / 1000 / 1000
        if model.device != device:
            deviceType = self.devices[device].type
            primDeviceType = self.devices[model.device].type
            totalTime += model.height_in * model.height_in * model.c_in * 4 / self.B[deviceType][primDeviceType] / 8 / 1024 / 1024 * 1000
        return totalTime

    def getDetailedTime(self, pop:list):
        deviceTimes = []
        for i in range(self.devices.__len__()):
            deviceTimes.append(0)
        computeTimes = 0
        transTimes = 0
        transFLOPS = 0
        for i in range(pop.__len__()):
            computeTime, transTime, modelTime = self.getTimeDetailed(pop[i],i)
            deviceTimes[pop[i]] += modelTime
            computeTimes += computeTime
            transTimes += transTime
            transFLOPS += transTime * (self.devices[pop[i]].p + self.devices[self.models[i].device].p)
        return computeTimes, transTimes, max(deviceTimes), transFLOPS

    def getTimeDetailed(self, device, index):
        model: MyModel = self.models[index]
        totalTime = 0
        for i in range(model.nodes.__len__()):
            totalTime += model.nodes[i].flops / self.devices[device].p / 1000 / 1000
        computeTime = totalTime
        transTime = 0
        if model.device != device:
            deviceType = self.devices[device].type
            primDeviceType = self.devices[model.device].type
            transTime += model.height_in * model.height_in * model.c_in * 4 / self.B[deviceType][primDeviceType] / 8 / 1024 / 1024 * 1000
            totalTime += transTime
        return computeTime, transTime, totalTime