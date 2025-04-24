from On_demand_Fine_grained_Partitioning.Device import Device


def construct_environment_yg():
    devices = [Device(0, 0, 4),
               Device(1, 1, 5),
               Device(2, 1, 6),
               Device(3, 2, 7)]
    bandwidth = [[6, 6, 6],
         [6, 6, 6],
         [6, 6, 6]]

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

def construct_environment_tg1():
    devices = [Device(0, 0, 8),
               Device(1, 1, 8),
               Device(2, 1, 8),
               Device(3, 2, 8)]
    bandwidth = [[16, 16, 16],
                 [16, 16, 16],
                 [16, 16, 16]]
    return devices, bandwidth

def construct_environment_real():
    devices = [Device(0, 0, 4),
               Device(1, 0, 8),
               Device(2, 1, 8),
               Device(3, 2, 24)]
    bandwidth = [[8, 8, 8],
                 [8, 8, 8],
                 [8, 8, 8]]
    return devices, bandwidth

def construct_environment_real1():
    devices = [Device(0, 0, 4),
               Device(1, 0, 8),
               Device(2, 1, 8)]
    bandwidth = [[8, 8, 8],
                 [8, 8, 8],
                 [8, 8, 8]]
    return devices, bandwidth

def construct_environment_real5():
    devices = [Device(0, 0, 14),
               Device(1, 0, 12.7),
               Device(2, 0, 12.5),
               Device(3, 1, 70),
               Device(4, 2, 90)]
    bandwidth = [[8, 8, 8],
                 [8, 8, 8],
                 [8, 8, 8]]
    return devices, bandwidth

def construct_environment_real7():
    devices = [Device(0, 0, 5),
               Device(1, 0, 5),
               Device(2, 0, 7),
               Device(3, 0, 7),
               Device(4, 1, 8),
               Device(5, 1, 8),
               Device(6, 2, 10)]
    bandwidth = [[8, 8, 8],
                 [8, 8, 8],
                 [8, 8, 8]]
    return devices, bandwidth

def construct_environment_reali():
    devices = [Device(0, 0, 6),
               Device(1, 1, 12),
               Device(2, 1, 9),
               Device(3, 2, 10)]
    bandwidth = [[10, 50, 80],
                 [10, 10, 30],
                 [30, 30, 10]]
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