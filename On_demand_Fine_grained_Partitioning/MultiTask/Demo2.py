from On_demand_Fine_grained_Partitioning.MultiTask import PSONoPartition, MyModel, MyParameters, TimeLineUtil, testMyPartition
from On_demand_Fine_grained_Partitioning import ConstructEnvironment, readJson
import copy
import time

# 我的方法OPM-MCNN
def exportModels(num):
    modelNum = num
    modelPools = [
        [0, "AlexNet", 227, 3],
        [0, "GoogleNet", 227, 3],
        [0, "Vgg16", 224, 3],
        [0, "ResNet50", 224, 3],
        [0, "YOLO", 448, 3]
    ]
    models = []
    for i in range(modelNum):
        # 选择模型的序号
        poolNumber = modelPools[3]
        # poolNumber = modelPools[i % 5]
        models.append(MyModel.MyModel(poolNumber[0],poolNumber[1],poolNumber[2],poolNumber[3]))

    sumFlops = 0
    for i in range(models[0].nodes.__len__()):
        print(i, models[0].nodes[i].flops)
        sumFlops += models[0].nodes[i].flops
    print("sumFlops", sumFlops)
    # 设置设备
    Device, B = ConstructEnvironment.construct_environment_real5()
    # 1. 无划分分配任务
    print("==========1. 无划分分配任务==========")
    now = time.time()
    testPSONoPartition = PSONoPartition.PSONoPartition('PSOGA', models, Device, B, MyParameters.max_pop_size, MyParameters.max_iter_size)
    best_dicision_sum = testPSONoPartition.run()
    BestDecision = best_dicision_sum[0]
    print(BestDecision)
    print(time.time() - now)
    # 2. 将任务残余量缩减至设备数量
    print("==========2. 将任务残余量缩减至设备数量==========")
    timeLines = TimeLineUtil.TimeLines(Device,B,models,BestDecision)
    timeLines.visitAll()
    print("等待时间和等待占用的FLOPS总共有:", timeLines.findDiffTime())
    # timeLines.decline()
    # timeLines.visitAll()
    # 3. 找到需要划分的任务和执行时间节点
    print("==========3. 找到需要划分的任务和执行时间节点==========")
    #minTime 最短时间
    #deviceList 模型对应的设备
    #modelList 模型
    #timeList 模型对应的结束时间
    minTime, deviceList, modelList, timeList = timeLines.findMaxModels()
    print(deviceList)
    maxTime = 0
    timeLines.visitAll()
    # 分布式执行一个最长时间结束模型，但不能影响别的设备执行
    while True:
        #找到最长时间结束的模型
        index = 0
        print(timeList, minTime)
        if timeList.__len__() == 0 or max(timeList) <= minTime:
            break
        for i in range(timeList.__len__()):
            if timeList[i] > timeList[index]:
                index = i
        model: MyModel.MyModel = modelList[index]
        device = deviceList[index]
        timeTmp = timeList[index]
        #从列表中删除模型
        modelList.pop(index)
        deviceList.pop(index)
        timeList.pop(index)
        #删除任务表
        timeLines.TimeLines[device].removeTaskLine()

        deviceTime = []
        #再找到所有设备可以执行该模型的时间点
        model.computeTime = max(0, minTime - model.startTime)
        for i in range(Device.__len__()):
            deviceTime.append(minTime)
        for i in range(timeList.__len__()):
            deviceTime[deviceList[i]] = max(deviceTime[deviceList[i]], timeList[i])
        print("目前所有设备空余时间列表", deviceTime)
        nodeIndex, computeTime, nowFlops = model.computePartModelWithTime(device, Device, B)
        deviceTime[device] = model.startTime + computeTime
        minTime = min(deviceTime)
        print("deviceTime",deviceTime,computeTime)
        for i in range(Device.__len__()):
            deviceTime[i] -= minTime
        print("模型目前计算的时间点和计算量",model.computeTime, nowFlops)
        timeLines.TimeLines[device].modifyTaskLine(nowFlops, 0, [device], [0])

        print("模型原长度", model.nodes.__len__())
        # 切割该模型
        model.fixPartModel(nodeIndex, device)
        print(model.name, "合并前", model.nodes.__len__(), nodeIndex, computeTime)
        model.nodes = readJson.combine_norm_relu(model.nodes)
        model.nodes = readJson.get_sort_nodes(readJson.combine_conv(model.nodes))
        print(model.name, "合并后", model.nodes.__len__())
        # 划分该模型
        print("划分模型前的时间",deviceTime, minTime)
        timeResult, decisionList = testMyPartition.run_fix_bp(Device, B, model, deviceTime, 0)


        minTime += timeResult[0]
        print("划分模型后的时间", minTime)
        maxTime = max(maxTime, minTime)
        print("deviceTime: ",deviceTime)
    print(minTime, min(best_dicision_sum[2]))
    print(time.time() - now)
    print(maxTime)
    return num, maxTime, min(best_dicision_sum[2])

if __name__ == '__main__':
    #模型数量
    exportModels(5)
    timeEfficiency = []
    resultList = []
    # for i in range(1, 52, 10):
    #     nowTime = time.time()
    #     resultList.append(exportModels(i))
    #     timeEfficiency.append(time.time() - nowTime)
    # for i in range(resultList.__len__()):
    #     print(i, resultList[i], timeEfficiency[i])



