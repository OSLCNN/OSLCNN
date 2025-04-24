import copy

import matplotlib.pyplot as plt
import pandas as pd
from On_demand_Fine_grained_Partitioning import ConstructEnvironment, Util, readJson, multiPartition, plotBar
from On_demand_Fine_grained_Partitioning.Device import Device
from On_demand_Fine_grained_Partitioning.multiPartition import ModelPro


def diff_cpu():     # 不同的带宽

    B = [[6, 6, 6],
         [6, 6, 6],
         [6, 6, 6]]
    # model_list = [ModelPro().AlexNet, ModelPro().GoogleNet, ModelPro().Vgg16, ModelPro().ResNet50]
    model_list = [ModelPro().YOLO]

    Util.B = B
    data = [[], [], [], [], [], []]
    # 加速比
    data_ratio = [[], [], [], [], [], []]
    data_label = ['Local', 'Central', 'PSOGA', 'EdgeLD', 'FPM', 'OFPM']
    writer = pd.ExcelWriter('output/diff_cpu_6.xls')
    cpu_list = [0, 20, 40, 60, 80]
    row = 0
    for model_name in model_list:
        df_time = pd.DataFrame(columns=cpu_list)
        df_ratio = pd.DataFrame(columns=cpu_list)
        for cpu in cpu_list:
            devices = [Device(0, 0, 8, 0),
                       Device(1, 1, 8, cpu),
                       Device(2, 1, 8, 0),
                       Device(3, 2, 8, 0)]
            Util.devices = devices
            nodes = readJson.construct_model(model_name, 448, 3)  # 指定模型，生成图结构
            nodes = readJson.combine_norm_relu(nodes)
            nodes = readJson.get_sort_nodes(readJson.combine_conv(nodes))
            name_list, time_list = multiPartition.run_fix_bp(model_name + "-" + str(cpu), devices, B,
                                                             nodes)
            print(time_list)
            # for i in range(len(time_list)):
            #     data[i].append(time_list[i])
            #     data_ratio[i].append(round(time_list[0] / time_list[i], 2))
            # df_time[cpu] = copy.deepcopy(time_list)
            # df_ratio[cpu] = [round(time_list[0] / time, 2) for time in time_list]  # 保留2位小数

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
    diff_cpu()
