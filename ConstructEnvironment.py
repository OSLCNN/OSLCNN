from On_demand_Fine_grained_Partitioning.Device import Device


def construct_environment_yg():
    devices = [Device(0, 0, 4),
               Device(1, 1, 5),
               Device(2, 1, 6),
               Device(3, 2, 7)]
    bandwidth = [[4, 4, 4],
         [4, 4, 4],
         [4, 4, 4]]

    return devices, bandwidth


def construct_environment_tg():
    devices = [Device(0, 0, 8),
               Device(1, 1, 8),
               Device(2, 1, 8),
               Device(3, 2, 8)]
    bandwidth = [[6, 6, 6],
                 [6, 6, 6],
                 [6, 6, 6]]
    return devices, bandwidth

def get_devices_1():
    devices = [Device(0, 0, 4),
               Device(1, 0, 4),
               Device(2, 1, 7),
               Device(3, 1, 8),
               Device(4, 2, 12)]
    return devices


def get_bandwidth_1():
    bandwidth = [[10, 10, 0.08],
                [10, 10, 0.5],
                [0.08, 0.5, 10]]
    return bandwidth