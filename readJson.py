# 根据json文件创建图
import json
import queue

from On_demand_Fine_grained_Partitioning.Node import Node, ConvNode, ReluNode, NormNode, PoolNode, DropNode, FcNode, ConcatNode, \
    CombineNode

new_nodes = []


def construct_model(model_name, height_in, c_in):
    file_name = model_name
    if not (len(model_name) > 5 and model_name[-5:] == ".json"):
        file_name = './json/' + model_name + '.json'
    with open(file_name, "r") as f:
        net = json.load(f)['nodes']
    nodes = []
    width_in = height_in
    for i, node_json in enumerate(net):
        node_id = node_json['id']
        node_type = node_json['type']
        node_name = node_json['name']
        parentList = node_json['parentList']
        childList = node_json['childList']
        if node_type == 'conv' or node_type == 'pool':      # 不同类型的层有各自不同的属性值
            k_size = node_json['k_size']
            if 'stride' in node_json.keys():
                stride = node_json['stride']
            else:
                stride = 1
            if 'padding' in node_json.keys():
                padding = node_json['padding']
            else:
                padding = 0
            if node_type == 'conv':
                k_num = node_json['k_num']
                nodes.append(ConvNode(node_id, node_type, node_name, k_size, k_num, stride, padding, parentList,
                                      childList))
            else:
                nodes.append(PoolNode(node_id, node_type, node_name, k_size, stride, padding, parentList, childList))
        elif node_type == 'relu':
            nodes.append(ReluNode(node_id, node_type, node_name, parentList, childList))
        elif node_type == 'norm':
            nodes.append(NormNode(node_id, node_type, node_name, parentList, childList))
        elif node_type == 'drop':
            nodes.append(DropNode(node_id, node_type, node_name, parentList, childList))
        elif node_type == 'fc':
            out = node_json['out']
            nodes.append(FcNode(node_id, node_type, node_name, out, parentList, childList))
        elif node_type == 'concat':
            nodes.append(ConcatNode(node_id, node_type, node_name, parentList, childList))

    nodes[0].set_in_out_shape(height_in, width_in, c_in)
    for index in nodes[0].child_list:
        nodes[0].child_nodes.append([nodes[index], nodes[0].height_out, nodes[0].width_out, nodes[0].c_out])

    for i in range(1, len(nodes)):
        c_out = 0
        for index in nodes[i].parent_list:
            c_out = c_out + nodes[index].c_out
            nodes[i].parent_nodes.append(
                [nodes[index], nodes[index].height_out, nodes[index].width_out, nodes[index].c_out])
        nodes[i].set_in_out_shape(nodes[i - 1].height_out, nodes[i - 1].width_out, c_out)
        for index in nodes[i].child_list:
            nodes[i].child_nodes.append([nodes[index], nodes[i].height_out, nodes[i].width_out, nodes[i].c_out])

    input_node = ConcatNode(-1, 'concat', "input", [], [0])
    input_node.height_out = height_in
    input_node.width_out = width_in
    input_node.c_out = c_in
    input_node.child_nodes.append([nodes[0], input_node.height_out, input_node.width_out, input_node.c_out])
    nodes[0].parent_nodes.append([input_node, input_node.height_out, input_node.width_out, input_node.c_out])
    nodes.insert(0, input_node)
    return nodes


# 将norm层和relu层与之前的节点合并，被合并的节点使用属性combine_nodes（list列表）保存在合并后的节点中
def combine_norm_relu(nodes_tmp):
    for node in nodes_tmp:
        if node.type == 'norm' or node.type == 'relu':
            # 只有一个父节点和一个尾节点
            if len(node.parent_nodes) == 1 and len(node.child_nodes) == 1:
                parent = node.parent_nodes[0][0]
                child = node.child_nodes[0][0]

                # 合并前：grandparent --- parent --- node --- child
                # 合并后：grandparent --- combine_node(parent, node) --- child
                if not isinstance(parent, CombineNode):  # 如果父节点不是CombineNode，就新建一个节点
                    combine_node = CombineNode(parent.id, "combine", parent.name)

                    # 建立 combine_node 到 grandparent 的连接（修改parent_nodes）(已经默认node只有一个parent)
                    combine_node.parent_nodes = parent.parent_nodes
                    # 建立 grandparent 到 combine_node 的连接(可能有多个grandparent)（修改child_nodes）
                    for grandparent_list in parent.parent_nodes:
                        for index, node_list in enumerate(grandparent_list[0].child_nodes):
                            if node_list[0] == parent:
                                break
                        grandparent_list[0].child_nodes[index][0] = combine_node
                    combine_node.child_nodes = parent.child_nodes
                    # 建立 combine_node 到 node 的连接（也就是先用combine_node代替parent的位置，之后的步骤再建立combine_node到child的连接）
                    combine_node.combine_nodes.append(parent)
                    if not combine_node.hasConv and isinstance(parent, ConvNode):
                        combine_node.hasConv = True
                else:  # 父节点已经是 CombineNode（说明已经合并过一个节点了）
                    combine_node = parent
                combine_node.combine_nodes.append(node)
                if not combine_node.hasConv and isinstance(node, ConvNode):
                    combine_node.hasConv = True

                combine_node.child_nodes = node.child_nodes  # 建立 combine_node 到 child 的连接(已经默认node只有一个child)

                for index, node_list in enumerate(child.parent_nodes):  # 建立 child 到 combine_node 的连接
                    if node_list[0] == node:
                        break
                child.parent_nodes[index][0] = combine_node

    return get_sort_nodes(nodes_tmp[0])


def combine_conv(nodes_tmp):
    # 合并连续的conv层
    for node in nodes_tmp:
        if isinstance(node, ConvNode) or node.hasConv:
            parent = node.parent_nodes[0][0]
            # grandparent --- parent --- node --- child 将node和parent合并为一个节点
            if isinstance(parent, ConvNode) or parent.hasConv:     # 将两个连续的conv层合并为一个
                child = node.child_nodes[0][0]
                # node为当前节点，combine_nodes是被合并的节点集合
                combine_nodes = [node]
                # 如果len(node.combine_nodes) > 0，说明当前节点已经是合并过后的节点了
                if len(node.combine_nodes) > 0:
                    combine_nodes = node.combine_nodes
                # 合并后的节点
                node_after_combine = parent
                if not isinstance(parent, CombineNode):     # 如果前面的节点不是Combine节点，需要新建一个
                    node_after_combine = CombineNode(0, 'combine', 'combine')
                    node_after_combine.hasConv = True
                    node_after_combine.combine_nodes.append(parent)
                    # 修改新节点和grandparent之间的连接
                    node_after_combine.parent_nodes = parent.parent_nodes
                    grandparent = parent.parent_nodes[0][0]        # conv层都只有一个输入和一个输出
                    for grandparent_child in grandparent.child_nodes:   # grandparent不一定是conv层，所以可能有多个输出
                        if grandparent_child[0] == parent:
                            grandparent_child[0] = node_after_combine
                # 将node节点包含的节点都合并到新的节点中去
                for com_node in combine_nodes:
                    node_after_combine.combine_nodes.append(com_node)
                # 修改新节点和child之间的连接
                node_after_combine.child_nodes = node.child_nodes
                for child_parent in child.parent_nodes:
                    if child_parent[0] == node:
                        child_parent[0] = node_after_combine
    return nodes_tmp[0]     # 返回的是第一个节点，如果要使用nodes列表，需要在调用后使用拓扑排序


def combine_branch(begin_node):
    # 合并每个分支中的所有节点
    current_node = begin_node
    while current_node:
        if len(current_node.child_nodes) == 0:
            break
        elif len(current_node.child_nodes) == 1:
            current_node = current_node.child_nodes[0][0]
        else:
            for index, child_nodes in enumerate(current_node.child_nodes):
                child_node = child_nodes[0]
                if len(child_node.parent_nodes) == 1:
                    new_node = CombineNode(0, 'combine', 'combine')
                    # 修改current_node --- new_node
                    current_node.child_nodes[index][0] = new_node
                    # 修改new_node --- current_node
                    new_node.parent_nodes = [[current_node, child_nodes[1], child_nodes[2], child_nodes[3]]]

                    before_node = child_node
                    while len(child_node.parent_nodes) == 1:
                        if len(child_node.combine_nodes) >= 1:
                            for node in child_node.combine_nodes:
                                new_node.combine_nodes.append(node)
                                if isinstance(node, ConvNode):
                                    new_node.hasConv = True
                        else:
                            if isinstance(child_node, ConvNode) or (isinstance(child_node, CombineNode) and child_node.hasConv):
                                new_node.hasConv = True
                            new_node.combine_nodes.append(child_node)
                        before_node = child_node
                        child_node = child_node.child_nodes[0][0]
                    # 修改new_node到最后
                    new_node.child_nodes = before_node.child_nodes
                    # 修改最后到new_node
                    for child_parent_list in child_node.parent_nodes:
                        if child_parent_list[0] == before_node:
                            child_parent_list[0] = new_node
            current_node = child_node
    nodes_res = get_sort_nodes(begin_node)
    return nodes_res


def get_sort_nodes(begin_node):
    q = queue.Queue()
    q.put(begin_node)
    visited_set = set()  # 集合，用来保存已经访问过的节点
    visited_set.add(begin_node)
    while not q.empty():  # 计算每个节点的入度
        next_node = q.get()
        # print(next_node.id)
        next_node.in_degree = len(next_node.parent_nodes)
        for child in next_node.child_nodes:
            if child[0] not in visited_set:
                # child.in_degree = 0
                q.put(child[0])
                visited_set.add(child[0])
    # 根据入度拓扑排序
    res = []
    q.put(begin_node)
    while not q.empty():
        next_node = q.get()
        res.append(next_node)
        for child in next_node.child_nodes:
            child[0].in_degree = child[0].in_degree - 1
            if child[0].in_degree == 0:
                q.put(child[0])
    return res


if __name__ == '__main__':
    # 测试方法是否可行
    test_nodes = construct_model("AlexNet", 227, 3)
    test_nodes = construct_model("ResNet50", 224, 3)
    test_nodes = construct_model("Vgg16", 227, 3)
    test_nodes = construct_model("GoogleNet", 227, 3)
    get_sort_nodes(test_nodes[0])
