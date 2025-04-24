import numpy as np

from On_demand_Fine_grained_Partitioning import parameters
from On_demand_Fine_grained_Partitioning.Node import ConvNode, FcNode, ConvConcatNode, CombineNode


class Device:

    def __init__(self, id, type, p, cpu=0):
        """
        type = 0:本地设备，1:边缘服务器，2:云服务器
        """
        self.id = id
        self.type = type
        self.p = p
        self.cpu = cpu
        self.period = []    # 执行时间段 start,end  # 先没考虑type = 0 : computing  1 : transmission

    def predict_time_by_node(self, node, flops, cpu=-1):

        def Pfun(t, a1, b1, a2, b2, c2, d2, e2, f2):
            return (a1 * t[0] + b1) * \
                   (a2 * np.power(t[1], 5) + b2 * np.power(t[1], 4) + c2 * np.power(t[1], 3) +
                    d2 * np.power(t[1], 2) + e2 * t[1] + f2)

        if cpu == -1:
            cpu = self.cpu
        xy = np.vstack((np.array([flops / 10 ** 6 * 1.0]), np.array([cpu * 1.0])))
        if isinstance(node, ConvNode) or (isinstance(node, ConvConcatNode)) or (isinstance(node, CombineNode) and node.hasConv):
            time = Pfun(xy, *parameters.conv_popt)[0] * 17
        elif isinstance(node, FcNode):
            time = Pfun(xy, *parameters.fc_popt)[0]
        else:
            time = flops / self.p / 1000 / 1000
        return time

    def predict_time_by_type(self, type, flops, cpu=-1):

        def Pfun(t, a1, b1, a2, b2, c2, d2, e2, f2):
            return (a1 * t[0] + b1) * \
                   (a2 * np.power(t[1], 5) + b2 * np.power(t[1], 4) + c2 * np.power(t[1], 3) +
                    d2 * np.power(t[1], 2) + e2 * t[1] + f2)
        if cpu == -1:
            cpu = self.cpu
        xy = np.vstack((np.array([flops / 10 ** 6 * 1.0]), np.array([cpu * 1.0])))
        if type == 'conv':
            time = Pfun(xy, *parameters.conv_popt)[0] * 17
        elif type == 'fc':
            time = Pfun(xy, *parameters.fc_popt)[0]
        else:
            time = flops / self.p / 1000 / 1000
        return time




