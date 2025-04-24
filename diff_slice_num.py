import copy

import matplotlib.pyplot as plt
import pandas as pd
from On_demand_Fine_grained_Partitioning import ConstructEnvironment, Util, readJson, multiPartition, plotBar, parameters
from On_demand_Fine_grained_Partitioning.Device import Device
from On_demand_Fine_grained_Partitioning.multiPartition import ModelPro


def diff_slice_shape():     # 不同的输入大小
    # devices, B = ConstructEnvironment.construct_environment_yg()  # 指定环境
    devices = [Device(0, 0, 4),
               Device(1, 1, 5),
               Device(2, 1, 6),
               Device(3, 2, 7)]
    B = [[6, 6, 6],
                 [6, 6, 6],
                 [6, 6, 6]]
    Util.devices = devices
    Util.B = B
    model_list = [ModelPro().AlexNet, ModelPro().GoogleNet, ModelPro().Vgg16, ModelPro().ResNet50]
    # model_list = [ModelPro().AlexNet, ModelPro().GoogleNet]
    slice_num_list = [0, 1, 2, 3, 4, 5, 6]
    data = [[], [], [], [], [], []]
    # 加速比
    data_ratio = [[], [], [], [], [], []]
    writer = pd.ExcelWriter('output/diff_slice.xls')
    df_time = pd.DataFrame(columns=slice_num_list)
    df_ratio = pd.DataFrame(columns=slice_num_list)
    row = 0
    for model_name in model_list:
        inner_data = []
        for slice_num in slice_num_list:
            print("模型为：" + str(model_name) + ", 形状为:" + str(slice_num))
            nodes = readJson.construct_model(model_name, 227, 3)  # 指定模型，生成图结构
            nodes = readJson.combine_norm_relu(nodes)
            nodes = readJson.get_sort_nodes(readJson.combine_conv(nodes))

            parameters.slice_num = slice_num
            # 只需要Local，FP和OFP，可以注释掉其他的代码
            name_list, time_list = multiPartition.run_fix_bp(model_name + "-" + str(slice_num), devices, B, nodes)
            # inner_data.append(time_list[-1])
            bar_label = name_list
            for i in range(len(time_list)):
                data[i].append(time_list[i])
                data_ratio[i].append(round(time_list[0] / time_list[i], 2))
            df_time[slice_num] = copy.deepcopy(time_list)
            df_ratio[slice_num] = [round(time_list[0] / time, 2) for time in time_list]  # 保留2位小数
        df_time.index = bar_label
        df_ratio.index = bar_label
        df_time.to_excel(writer, startrow=row, startcol=0)
        df_ratio.to_excel(writer, startrow=row, startcol=13)
        row += 10
    writer.save()

    # plt.figure()
    # plotBar.create_multi_bars(data, '不同划分数不同模型的时延', [str(s) for s in slice_num_list], '时延（ms）', model_list)
    # plt.show()


if __name__ == '__main__':
    diff_slice_shape()
