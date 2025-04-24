import random

from On_demand_Fine_grained_Partitioning import parameters
from On_demand_Fine_grained_Partitioning import ConstructEnvironment
class convNodeTime:

    def __init__(self,deviceNum):
        #设备的总数
        self.devideResult = []
        self.runResult = []
        for i in range(deviceNum):
            temp1 = []
            for j in range(deviceNum):
                temp2 = []
                for k in range(deviceNum):
                    temp2.append(random.randint(1,10))
                temp1.append(temp2)
            self.runResult.append(temp1)
        for i in range(deviceNum):
            temp1 = []
            for j in range(deviceNum):
                temp1.append(0)
            self.devideResult.append(temp1)

    def check(self,L,R,num):
        print(self.runResult)
        self.runResult[L][R].sort(reverse=True)
        print(self.runResult[L][R])


    def append(self,L,R,time):
        for i in range(time.__len__()):
            self.runResult[L][R][i] = time[i]
        self.runResult[L][R].sort(reverse=True)

    def getTime(self,L,R,slice_num,node):
        if self.devideResult[L][R] == 0:
            self.append(L,R)
        temp = []
        for i in range(slice_num):
            sum = self.runResult[L][R][i]
        for i in range(slice_num):
            temp.append(self.runResult[L][R][i]/sum)


        H = node.height_out
        for i in range(slice_num):
            sum
        res = []



x1,x2 = ConstructEnvironment.construct_environment_yg()
myConv = convNodeTime(2,x1.__len__())
myConv.check(1,2,2)
