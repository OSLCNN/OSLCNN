import copy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from brokenaxes import brokenaxes

from On_demand_Fine_grained_Partitioning import ConstructEnvironment, Util, readJson, multiPartition, plotBar, parameters
from On_demand_Fine_grained_Partitioning.Device import Device
from On_demand_Fine_grained_Partitioning.multiPartition import ModelPro


def diff_model():     # 不同的输入大小
    devices = [Device(0, 0, 4),
               Device(1, 1, 5),
               Device(2, 1, 6),
               Device(3, 2, 7)]
    B = [[10, 10, 10],
                 [10, 10, 10],
                 [10, 10, 10]]
    Util.devices = devices
    Util.B = B
    model_name_list = [ModelPro().AlexNet, ModelPro().GoogleNet, ModelPro().Vgg16, ModelPro().ResNet50]
    height = 227
    # data = [[], [], [], [], [], []]
    # data_ratio = [[], [], [], [], [], []]
    bar_label = ['Local', 'Central', 'PSOGA', 'EdgeLD', 'FPM', 'OFPM']
    writer = pd.ExcelWriter('output/diff_model.xls')
    df_time = pd.DataFrame(columns=(ModelPro().AlexNet, ModelPro().GoogleNet, ModelPro().Vgg16, ModelPro().ResNet50))
    df_ratio = pd.DataFrame(columns=(ModelPro().AlexNet, ModelPro().GoogleNet, ModelPro().Vgg16, ModelPro().ResNet50))
    # row = 0
    for model_name in model_name_list:

        if model_name == ModelPro().Vgg16 or model_name == ModelPro().ResNet50:
            height = 224
        nodes = readJson.construct_model(model_name, height, 3)  # 指定模型，生成图结构
        nodes = readJson.combine_norm_relu(nodes)
        nodes = readJson.get_sort_nodes(readJson.combine_conv(nodes))

        name_list, time_list = multiPartition.run_fix_bp(model_name, devices, B, nodes)

        df_time[model_name] = copy.deepcopy(time_list)
        df_ratio[model_name] = [round(time_list[0] / time, 2) for time in time_list]    # 保留2位小数
    df_time.index = bar_label
    df_ratio.index = bar_label

    df_time.to_excel(writer)
    df_ratio.to_excel(writer, startrow=0, startcol=8)
    writer.save()
    data = []
    for model_name in model_name_list:
        data.append(np.array(df_time[model_name]).tolist())
    data2 = []  # 交换行列
    for i in range(len(data[0])):
        b = []
        for j in range(len(data)):
            b.append(data[j][i])
        data2.append(b)
    print(data2)
    plt.figure()
    plotBar.create_multi_bars(data2, '不同算法关于不同模型的对比', model_name_list, '时延(ms)', bar_label)
    plt.show()


if __name__ == '__main__':
    diff_model()
