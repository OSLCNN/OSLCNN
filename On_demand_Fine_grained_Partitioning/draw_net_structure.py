from PartitionTest import partition
import matplotlib.pyplot as plt


def draw(name, height_in):
    partition.pre_partition(name, height_in, 3)
    plt.figure(1)
    partition.net_structure()
    plt.figure(2)
    partition.net_latency()


if __name__ == '__main__':
    # # # # # # # # # AlexNet # # # # # # # # #
    # draw("AlexNet", 227)
    # # # # # # # # # GoogleNet # # # # # # # # #
    # draw("GoogleNet", 227)
    # # # # # # # # # Vgg16 # # # # # # # # #
    # draw("Vgg16", 224)
    # # # # # # # # # ResNet50 # # # # # # # # #
    draw("ResNet50", 224)
