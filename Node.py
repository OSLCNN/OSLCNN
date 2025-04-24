"""
节点类
"""
import math


class Node:
    def __init__(self, id, type, name, parent_list=[], child_list=[]):
        self.id = int(id)
        self.type = type
        self.name = name
        self.parent_list = parent_list
        self.child_list = child_list
        self.beginTime = 0
        self.finishTime = 0
        self.vmId = 0  # 执行位置
        self.height_in = 0
        self.width_in = 0       # 一般来说，输入和输出都是正方形矩阵，使用width是为了之后的切割
        self.c_in = 0
        self.height_out = 0
        self.width_out = 0
        self.c_out = 0
        self.flops = 0
        self.k_size = 0
        self.k_num = 0
        self.stride = 0
        self.padding = 0
        self.combine_nodes = []     # 被合并在一起的节点
        self.parent_nodes = []     # 格式：[[Node, height_out, width_out, c_out],[...]]
        self.child_nodes = []
        self.in_degree = -1
        self.visible = True     # 表示当前节点是否可见（是否被合并）
        self.is_first = True
        self.is_last = True
        self.hasConv = False
        self.need_assign = False    # True表示该节点为水平分割后进行合并的节点，需要进行按需分配
        # self.is_partitioned = False     # True 表示该节点是将conv层进行水平分割后的一个子节点

    def set_in_out_shape(self, height_in, width_in, c_in):
        # 指定输入形状，设置输入和输出的形状（默认输出与输入一样，若不一样，在子类中自行修改）
        self.height_out = self.height_in = height_in
        self.width_out = self.width_in = width_in
        self.c_out = self.c_in = c_in

    # def set_height_width_c_out(self, height_out, width_out, c_out):
    #     self.height_out = height_out
    #     self.width_out = width_out
    #     self.c_out = c_out

    def print_node(self):
        print("id: " + str(self.id) +
              ", name: " + str(self.name) +
              ", type: " + str(self.type) +
              ", height_in: " + str(self.height_in) +
              ", width_in: " + str(self.width_in) +
              ", c_in: " + str(self.c_in) +
              ", height_out: " + str(self.height_in) +
              ", width_out: " + str(self.width_out) +
              ", c_out: " + str(self.c_out) +
              ", flops:" + str(self.flops))


class ConvNode(Node):
    def __init__(self, id, type, name, k_size, k_num, stride, padding, parent_list, child_list):
        Node.__init__(self, id, type, name, parent_list, child_list)
        self.k_size = k_size
        self.k_num = k_num
        self.stride = stride
        self.padding = padding
        self.hasConv = True

    def set_in_out_shape(self, height_in, width_in, c_in):
        Node.set_in_out_shape(self, height_in, width_in, c_in)
        self.width_out = self.height_out = math.ceil((height_in - self.k_size + 2 * self.padding) / self.stride + 1)
        self.c_out = self.k_num
        self.flops = 2 * (self.height_in+2*self.padding) * (self.width_in+2*self.padding) * (self.c_in * self.k_size * self.k_size + 1) * self.c_out


class ReluNode(Node):
    def __init__(self, id, type, name, parent_list, child_list):
        Node.__init__(self, id, type, name, parent_list, child_list)
        self.flops = 0


class NormNode(Node):
    def __init__(self, id, type, name, parent_list, child_list):
        Node.__init__(self, id, type, name, parent_list, child_list)
        self.flops = 0


class PoolNode(Node):
    def __init__(self, id, type, name, k_size, stride, padding, parent_list, child_list):
        Node.__init__(self, id, type, name, parent_list, child_list)
        self.k_size = k_size
        self.stride = stride
        self.padding = padding

    def set_in_out_shape(self, height_in, width_in, c_in):
        Node.set_in_out_shape(self, height_in, width_in, c_in)
        self.width_out = self.height_out = math.ceil((height_in - self.k_size + 2 * self.padding) / self.stride + 1)
        self.flops = self.k_size * self.k_size * self.height_out * self.width_out * self.c_in


class DropNode(Node):
    def __init__(self, id, type, name, parent_list, child_list):
        Node.__init__(self, id, type, name, parent_list, child_list)
        self.flops = 2000


class FcNode(Node):
    def __init__(self, id, type, name, out, parent_list, child_list):
        Node.__init__(self, id, type, name, parent_list, child_list)
        self.out = out

    def set_in_out_shape(self, height_in, width_in, c_in):
        Node.set_in_out_shape(self, height_in, width_in, c_in)
        self.height_out = 1
        self.width_out = 1
        self.c_out = self.out
        self.flops = (2 * height_in * width_in * c_in - 1) * self.c_out


class ConcatNode(Node):
    def __init__(self, id, type, name, parent_list, child_list):
        Node.__init__(self, id, type, name, parent_list, child_list)

    # def set_in_out_shape(self, height_in, width_in, c_in):
    #     Node.set_in_out_shape(self, height_in, width_in, c_in)


class ConvConcatNode(Node):     # 用来表示当前节点是经过水平切割后的卷积层
    def __init__(self, id, type, name, k_size, k_num, stride, padding):
        Node.__init__(self, id, type, name)
        self.k_size = k_size
        self.k_num = k_num
        self.stride = stride
        self.padding = padding

    # def set_in_out_shape(self, height_in, width_in, c_in):
    #     Node.set_in_out_shape(self, height_in, width_in, c_in)


# class CombineNode(Node):     # 用来表示当前节点是合并后的节点
#     def __init__(self, id, type, name, k_size, k_num, stride, padding, parent_list, child_list):
#         Node.__init__(self, id, type, name, parent_list, child_list)
#         self.k_size = k_size
#         self.k_num = k_num
#         self.stride = stride
#         self.padding = padding
#         self.hasConv = False        # 合并后的节点是否包含卷积层
#         # self.need_assign = True


class CombineNode(Node):     # 用来表示当前节点是合并后的节点
    def __init__(self, id, type, name):
        Node.__init__(self, id, type, name)
        # self.k_size = k_size
        # self.k_num = k_num
        # self.stride = stride
        # self.padding = padding
        # self.hasConv = False        # 合并后的节点是否包含卷积层
        # self.need_assign = True
