import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from On_demand_Fine_grained_Partitioning import ConstructEnvironment, Util, readJson, multiPartition, plotBar
from On_demand_Fine_grained_Partitioning.Device import Device
from On_demand_Fine_grained_Partitioning.multiPartition import ModelPro
from brokenaxes import brokenaxes


def diff_input_shape():     # 不同的输入大小
    devices = [Device(0, 0, 8),
               Device(1, 1, 8),
               Device(2, 1, 8),
               Device(3, 2, 8)]
    B = [[6, 6, 6],
         [6, 6, 6],
         [6, 6, 6]]
    Util.devices = devices
    Util.B = B
    # model_list = [ModelPro().AlexNet, ModelPro().GoogleNet, ModelPro().Vgg16, ModelPro().ResNet50]
    model_list = [ModelPro().AlexNet, ModelPro().GoogleNet]
    shape_list = [24, 64, 112, 224, 448]
    data = [[], [], [], [], [], []]
    # 加速比
    data_ratio = [[], [], [], [], [], []]
    # model_name = ModelPro().AlexNet
    bar_label = []
    x = range(len(shape_list))

    writer = pd.ExcelWriter('output/diff-input-shape.xls')
    df_time = pd.DataFrame(columns=(24, 64, 112, 224, 448))
    df_ratio = pd.DataFrame(columns=(24, 64, 112, 224, 448))
    row = 0
    for model_name in model_list:
        for shape in shape_list:
            print("模型为：" + str(model_name) + ", 形状为:" + str(shape))
            nodes = readJson.construct_model(model_name, shape, 3)  # 指定模型，生成图结构
            nodes = readJson.combine_norm_relu(nodes)
            nodes = readJson.get_sort_nodes(readJson.combine_conv(nodes))
            name_list, time_list = multiPartition.run_fix_bp(model_name + "-" + str(shape), devices, B, nodes)
            bar_label = name_list
            for i in range(len(time_list)):
                data[i].append(time_list[i])
                data_ratio[i].append(round(time_list[0] / time_list[i], 2))
            df_time[shape] = copy.deepcopy(time_list)
            df_ratio[shape] = [round(time_list[0] / time, 2) for time in time_list]  # 保留2位小数
        df_time.index = bar_label
        df_ratio.index = bar_label
        df_time.to_excel(writer, startrow=row, startcol=0)
        df_ratio.to_excel(writer, startrow=row, startcol=13)
        row += 10
    writer.save()

    # plt.figure()
    # for inner_data in data_ratio:
    #     plt.plot(x, inner_data, marker='o', markersize=3)
    # plt.ylabel('加速比(x)')
    # plt.xlabel('输入特征图大小')
    # plt.xticks(x, [str(b) for b in shape_list])
    # plt.legend(bar_label)  # 设置折线名称
    # plt.show()

    # for model_name in model_list:
    #     inner_data = []
    #     for shape in shape_list:
    #         print("模型为：" + str(model_name) + ", 形状为:" + str(shape))
    #         # plt.figure()
    #         nodes = readJson.construct_model(model_name, shape, 3)  # 指定模型，生成图结构
    #         # 1. 合并norm层和relu层
    #         nodes = readJson.combine_norm_relu(nodes)
    #         # 2. 合并连续的conv层
    #         nodes = readJson.get_sort_nodes(readJson.combine_conv(nodes))
    #
    #         name_list, time_list = multiPartition.run_fix_bp(model_name + "-" + str(shape), devices, B, nodes)
    #         inner_data.append(time_list[-1])
    #         i += 1
    #     data.append(inner_data)

    # plt.figure()
    # bax = brokenaxes(ylims=((0, 750), (1250, 1400), (1850, 1950), (6750, 7050)), hspace=0.05, despine=False)
    #
    # tick_step = 1
    # group_gap = 0.2
    # bar_gap = 0
    # x = np.arange(len([str(s) for s in shape_list])) * tick_step
    # group_num = len(data)
    # group_width = tick_step - group_gap
    # bar_span = group_width / group_num
    # bar_width = bar_span - bar_gap
    # for index, y in enumerate(data):
    #     bax.bar(x + index * bar_span, y, bar_width, label=model_list[index])
    # bax.set_ylabel('时延（ms）')
    # bax.set_title('不同输入不同模型的时延')
    # # ticks = x + (group_width - bar_span) / 2
    # bax.legend()
    # plt.show()


if __name__ == '__main__':
    diff_input_shape()
