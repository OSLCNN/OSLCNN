import numpy as np
import matplotlib.pyplot as plt


def create_multi_bars(datas, title, x_labels, y_label, bar_labels, tick_step=1, group_gap=0.2, bar_gap=0):
    '''
      labels : x轴坐标标签序列
      datas ：数据集，二维列表，要求列表每个元素的长度必须与labels的长度一致
      tick_step ：默认x轴刻度步长为1，通过tick_step可调整x轴刻度步长。
      group_gap : 柱子组与组之间的间隙，最好为正值，否则组与组之间重叠
      bar_gap ：每组柱子之间的空隙，默认为0，每组柱子紧挨，正值每组柱子之间有间隙，负值每组柱子之间重叠
      '''
    # x为每组柱子x轴的基准位置
    x = np.arange(len(x_labels)) * tick_step
    # group_num为数据的组数，即每组柱子的柱子个数
    group_num = len(datas)
    # group_width为每组柱子的总宽度，group_gap 为柱子组与组之间的间隙。
    group_width = tick_step - group_gap
    # bar_span为每组柱子之间在x轴上的距离，即柱子宽度和间隙的总和
    bar_span = group_width / group_num
    # bar_width为每个柱子的实际宽度
    bar_width = bar_span - bar_gap
    # 绘制柱子
    for index, y in enumerate(datas):
        plt.bar(x + index * bar_span, y, bar_width, label=bar_labels[index])
    plt.ylabel(y_label)
    plt.title(title)
    # ticks为新x轴刻度标签位置，即每组柱子x轴上的中心位置
    ticks = x + (group_width - bar_span) / 2
    # plt.xticks(ticks, x_labels, rotation='vertical')
    plt.xticks(ticks, x_labels)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    size = 4
    x = np.arange(size)
    a = np.random.random(size)
    b = np.random.random(size)
    c = np.random.random(size)
    d = np.random.random(size)
    data = [a,b,c,d]
    bar_labels = ['bar1', 'bar2', 'bar3', 'bar4']
    x_labels = ['x1', 'x2', 'x3', 'x4']
    title = 'multi bars'
    y_label = 'y_label'
    create_multi_bars(data, title, x_labels, y_label, bar_labels)