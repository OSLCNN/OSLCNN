from On_demand_Fine_grained_Partitioning import parameters
from On_demand_Fine_grained_Partitioning.Util import get_series_height_in
import numpy as np
from On_demand_Fine_grained_Partitioning.Node import CombineNode, ConvNode, FcNode, PoolNode, ConvConcatNode, ConcatNode

def on_demand_assignment_adv(decision,nodes):
    decision = np.insert(decision, 0, 0)

    # decision 和 nodes 的顺序是一致的
    for i, begin_node in enumerate(nodes):
        # print(i,begin_node)
        if begin_node.need_assign:  # 根据首结点来确定需要划分的节点的组合

            # begin_device = devices[decision[i]]
            begin_node_child_nodes = begin_node.child_nodes
            begin_node.vmId = decision[i]
            branch_nodes = []  # 每个分支对应几个节点
            index = i
            child_i = 0
            device_id_visit = []  # 设备号，对应于device_nodes列表中的每一行

#针对每个孩子节点
            while child_i < len(begin_node_child_nodes):
                child_node = begin_node_child_nodes[child_i][0]
                if not isinstance(child_node, ConvConcatNode):#孩子不是卷积层
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
            #print(i,length)
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


