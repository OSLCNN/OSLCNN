from On_demand_Fine_grained_Partitioning import readJson
from On_demand_Fine_grained_Partitioning.Node import Node, ConcatNode
import copy
class MyModel:
    def __init__(self,device,name,height_in,c_in):
        #模型的节点
        self.nodes = readJson.construct_model(name,height_in,c_in,"../")
        self.name = name
        #模型从属于哪个设备
        self.device = device
        #模型总共的FLOPS
        self.FLOPS = self.computeFLOPS()
        self.height_in = height_in
        self.c_in = c_in
        # 当前模型执行的时间
        self.computeTime = 0
        # print(self.FLOPS)
        self.startTime = 0
        self.endTime = 0

    def computeFLOPS(self):
        flops = 0
        for node in self.nodes:
            flops += node.flops
        return flops

    #找到当前模型执行的时间对应的节点
    def computePartModelWithTime(self, index, Device, B: list):
        time = 0
        if index != self.device:
            deviceType = Device[index].type
            primDeviceType = Device[self.device].type
            time += self.height_in * self.height_in * self.c_in * 4 / B[deviceType][primDeviceType] / 8 / 1024 / 1024 * 1000
        # 如果还未开始传输，则返回-1
        if self.computeTime <= time and time != 0:
            return -1, 0, 0
        flops = 0
        for i in range(self.nodes.__len__()):
            node: Node = self.nodes[i]
            interval = node.flops / Device[index].p / 1000 / 1000
            flops += node.flops
            if self.computeTime >= time and self.computeTime < time + interval:
                # 倒退时间为执行该层前
                self.computeTime = time
                return i, self.computeTime, flops
            time += interval

    #组合成一个健康的模型
    def fixPartModel(self, index, device):
        #总体思路： 需要建立一个虚拟节点，这个节点没有计算量，仅充当起点，连接到截断节点上
        #1. 虚拟节点需要连接到截断节点的起点，这里在DAG图中可能会出现两个起点的情况
        if index == -1:
            return
        self.device = device
        self.nodes = self.nodes[index: self.nodes.__len__()]
        nowNode = []
        input_node = ConcatNode(-1, 'concat', "input", [], [])
        #2. 找到多个起点的方案是，找到截断后没有父亲节点的节点
        for i in range(self.nodes.__len__()):
            node: Node = self.nodes[i]
            if node.parent_nodes.__len__() == 0:
                nowNode.append(node)
                continue
            newtemp = []
            add = True
            for j in range(node.parent_nodes.__len__()):
                if node.parent_nodes[j][0].id + 1 >= index:
                    newtemp.append(node.parent_nodes[j])
                else:
                    if add:
                        nowNode.append(node)
                        add = False
            node.parent_nodes = copy.copy(newtemp)
        node: Node = self.nodes[0]
        input_node.height_out = node.height_in
        input_node.width_out = node.width_in
        input_node.c_out = node.c_in
        for i in range(nowNode.__len__()):
            input_node.child_nodes.append([nowNode[i], input_node.height_out, input_node.width_out, input_node.c_out])
            nowNode[i].parent_nodes.append([input_node, input_node.height_out, input_node.width_out, input_node.c_out])
        self.nodes.insert(0, input_node)
        print("切割",self.nodes.__len__())

if __name__ == '__main__':
    testModel2 = MyModel(1, "AlexNet", 227, 3)
    testModel4 = MyModel(1, "GoogleNet", 227, 3)
    testModel5 = MyModel(1, "Vgg16", 224, 3)
    testModel3 = MyModel(1, "ResNet50", 224, 3)
    testModel1 = MyModel(1, "YOLO", 448, 3)




