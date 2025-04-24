import copy

import matplotlib.pyplot as plt
import pandas as pd
from On_demand_Fine_grained_Partitioning import ConstructEnvironment, Util, readJson, multiPartition, plotBar, parameters
from On_demand_Fine_grained_Partitioning.Device import Device
from On_demand_Fine_grained_Partitioning.multiPartition import ModelPro


def diff_device_num():     # 不同的设备数量（同构）
    # devices = [Device(0, 0, 4),
    #            Device(1, 1, 7),
    #            Device(2, 1, 7),
    #            Device(3, 1, 7),
    #            Device(4, 2, 12)]
    # bandwidth = [[10, 10, 0.08],
    #              [10, 10, 0.5],
    #              [0.08, 0.5, 10]]

    device_num_list = [1, 2, 3, 4, 5, 6]
    # device_num_list = [3, 4, 5, 6, 7, 8, 9, 10]
    model_name_list = [ModelPro().AlexNet]
    # model_name_list = [ModelPro().AlexNet, ModelPro().GoogleNet]
    # model_name_list = [ModelPro().AlexNet, ModelPro().GoogleNet, ModelPro().Vgg16, ModelPro().ResNet50]
    data_label = ['Local', 'Central', 'PSOGA', 'EdgeLD', 'FPM', 'OFPM']
    df_time = pd.DataFrame(columns=device_num_list)
    df_ratio = pd.DataFrame(columns=device_num_list)

    x = range(len(device_num_list))
    writer = pd.ExcelWriter('output/diff_device-num.xls')
    row = 0
    for model_name in model_name_list:
        data = [[], [], [], [], [], []]
        data_ratio = [[], [], [], [], [], []]   # 加速比
        if model_name == "AlexNet":
            bandwidth = [[4, 4, 4],
                         [4, 4, 4],
                         [4, 4, 4]]
            Util.B = bandwidth
        elif model_name == "GoogleNet":
            bandwidth = [[4, 4, 4],
                         [4, 4, 4],
                         [4, 4, 4]]
            Util.B = bandwidth
        for device_num in device_num_list:
            # devices = [Device(0, 0, 5)]
            devices = []
            # for i in range(device_num):
            #     devices.append(Device(0, 0, 8))
            if device_num == 1:
                devices = [Device(0, 0, 8)]
            elif device_num == 2:
                devices = [Device(0, 0, 8), Device(1, 1, 8)]
            elif device_num == 3:
                devices = [Device(0, 0, 8), Device(1, 1, 8), Device(2, 2, 8)]
            elif device_num == 4:
                devices = [Device(0, 0, 8), Device(1, 1, 8), Device(2, 1, 8), Device(3, 2, 8)]
            elif device_num == 5:
                devices = [Device(0, 0, 8), Device(1, 0, 8), Device(2, 1, 8), Device(3, 1, 8), Device(4, 2, 8)]
            elif device_num == 6:
                devices = [Device(0, 0, 8), Device(1, 0, 8), Device(2, 1, 8), Device(3, 1, 8), Device(4, 2, 8), Device(5, 2, 8)]
            parameters.slice_num = device_num
            # parameters.slice_num = 5
            # devices.append(Device(4, 2, 12))
            Util.devices = devices
            nodes = readJson.construct_model(model_name, 227, 3)  # 指定模型，生成图结构
            nodes = readJson.combine_norm_relu(nodes)
            nodes = readJson.get_sort_nodes(readJson.combine_conv(nodes))
            name_list, time_list = multiPartition.run_fix_bp(model_name + "-设备数量" + str(device_num), devices, bandwidth,
                                                             nodes)
            for i in range(len(time_list)):
                data[i].append(time_list[i])
                data_ratio[i].append(round(time_list[0] / time_list[i], 2))

            df_time[device_num] = copy.deepcopy(time_list)
            df_ratio[device_num] = [round(time_list[0] / time, 2) for time in time_list]  # 保留2位小数
        df_time.index = data_label
        df_ratio.index = data_label
        df_time.to_excel(writer, startrow=row, startcol=0)
        df_ratio.to_excel(writer, startrow=row, startcol=13)

        print("执行时间" + str(df_time))
        print("加速比" + str(df_ratio))

        row += 10
        # plt.figure()
        # for inner_data in data:
        #     plt.plot(x, inner_data, marker='o', markersize=3)
        # plt.ylabel('时延（ms）')
        # plt.xlabel('设备数量')
        # plt.title(model_name)
        # plt.xticks(x, [str(b) for b in device_num_list])
        # plt.legend(data_label)  # 设置折线名称
        # plt.show()
    writer.save()




if __name__ == '__main__':
    diff_device_num()
