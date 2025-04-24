import copy

import matplotlib.pyplot as plt
import pandas as pd
from On_demand_Fine_grained_Partitioning import ConstructEnvironment, Util, readJson, multiPartition, plotBar, parameters
from On_demand_Fine_grained_Partitioning.Device import Device
from On_demand_Fine_grained_Partitioning.multiPartition import ModelPro


def diff_bandwidth_cpu():     # 不同的带宽,cpu
    # devices = [Device(0, 0, 4),
    #            Device(1, 0, 4),
    #            Device(2, 1, 7),
    #            Device(3, 1, 7),
    #            Device(4, 2, 12)]
    # devices = [Device(0, 0, 8),
    #            Device(1, 1, 8),
    #            Device(2, 1, 8),
    #            Device(3, 2, 8)]
    # model_list = [ModelPro().AlexNet, ModelPro().GoogleNet, ModelPro().Vgg16, ModelPro().ResNet50]
    # model_list = [ModelPro().AlexNet, ModelPro().GoogleNet]
    model_list = [ModelPro().GoogleNet]


    # Util.devices = devices
    data = [[], [], [], [], [], []]
    # 加速比
    data_ratio = [[], [], [], [], [], []]
    data_label = ['Local', 'Central', 'PSOGA', 'EdgeLD', 'FPM', 'OFPM']
    # x = range(len(edge_bandwidth_list))
    # model_name = ModelPro().AlexNet
    # writer = pd.ExcelWriter('output/diff_bandwidth-' + model_name + '.xls')
    writer = pd.ExcelWriter('output/diff_bandwidth.xls')
    row = 0
    for model_name in model_list:
        if model_name == 'AlexNet':
            edge_bandwidth_list = [0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6]
        else:
            edge_bandwidth_list = [0.5, 2, 4, 6, 8, 10, 12, 14, 16]
        df_time = pd.DataFrame(columns=edge_bandwidth_list)
        df_ratio = pd.DataFrame(columns=edge_bandwidth_list)
        cpu_list = [0, 20, 40, 60, 80]
        for cpu in cpu_list:
            # parameters.cpu = cpu
            devices = [Device(0, 0, 8, cpu),
                       Device(1, 1, 8, cpu),
                       Device(2, 1, 8, cpu),
                       Device(3, 2, 8, cpu)]
            Util.devices = devices
            for edge_bandwidth in edge_bandwidth_list:
                bandwidth = [[edge_bandwidth, edge_bandwidth, edge_bandwidth],
                             [edge_bandwidth, edge_bandwidth, edge_bandwidth],
                             [edge_bandwidth, edge_bandwidth, edge_bandwidth]]
                # bandwidth[0][1] = bandwidth[1][0] = bandwidth[0][0] = bandwidth[1][1] = edge_bandwidth
                Util.B = bandwidth
                nodes = readJson.construct_model(model_name, 227, 3)  # 指定模型，生成图结构
                nodes = readJson.combine_norm_relu(nodes)
                nodes = readJson.get_sort_nodes(readJson.combine_conv(nodes))
                # name_list = ['Edge-Only', 'Cloud-Only', 'CBPS', 'PSOGA', 'Algorithm1', 'Algorithm2']
                name_list, time_list = multiPartition.run_fix_bp(model_name + "-" + str(edge_bandwidth), devices,
                                                                 bandwidth, nodes)
                for i in range(len(time_list)):
                    data[i].append(time_list[i])
                    data_ratio[i].append(round(time_list[0] / time_list[i], 2))
                df_time[edge_bandwidth] = copy.deepcopy(time_list)
                df_ratio[edge_bandwidth] = [round(time_list[0] / time, 2) for time in time_list]  # 保留2位小数

            df_time.index = data_label
            df_ratio.index = data_label
            df_time.to_excel(writer, startrow=row, startcol=0)
            df_ratio.to_excel(writer, startrow=row, startcol=13)

            row += 13

    writer.save()

    # plt.figure()
    # for inner_data in data_ratio:
    #     plt.plot(x, inner_data, marker='o', markersize=3)
    # plt.ylabel('时延（ms）')
    # plt.xlabel('带宽（Mbps）')
    # plt.xticks(x, [str(b) for b in edge_bandwidth_list])
    # plt.legend(data_label)  # 设置折线名称
    # plt.show()


if __name__ == '__main__':
    diff_bandwidth_cpu()
