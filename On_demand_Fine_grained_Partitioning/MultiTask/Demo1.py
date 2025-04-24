from On_demand_Fine_grained_Partitioning.MultiTask import PSONoPartition, MyModel, MyParameters, TimeLineUtil, testMyPartition
from On_demand_Fine_grained_Partitioning import ConstructEnvironment, readJson
import copy

def exportModels():
    modelNum = 5
    modelPools = [
        [0, "AlexNet", 227, 3],
        [0, "GoogleNet", 227, 3],
        [0, "Vgg16", 224, 3],
        [0, "ResNet50", 224, 3],
        [0, "YOLO", 448, 3]
    ]
    models = []
    for i in range(modelNum):
        poolNumber = modelPools[4]
        # poolNumber = modelPools[i % 5]
        models.append(MyModel.MyModel(poolNumber[0],poolNumber[1],poolNumber[2],poolNumber[3]))

    sumFLOPS = 0
    for i in range(models[0].nodes.__len__()):
        print(i, models[0].nodes[i].flops)
        sumFLOPS += models[0].nodes[i].flops

    Device, B = ConstructEnvironment.construct_environment_real()
    timeCloud = sumFLOPS / Device[3].p / 1000 / 1000 * modelNum + models[0].height_in * models[0].height_in * models[0].c_in * 4 / B[0][0] / 8 / 1024 / 1024 * 1000 * modelNum
    print("云端运行时间", timeCloud)
    return



    # 1. 无划分分配任务
    print("==========1. 无划分分配任务==========")
    testPSONoPartition = PSONoPartition.PSONoPartition('PSOGA', models, Device, B, MyParameters.max_pop_size, MyParameters.max_iter_size)
    best_dicision_sum = testPSONoPartition.run()
    BestDecision = best_dicision_sum[0]
    print(BestDecision)
    # 2. 将任务残余量缩减至设备数量
    print("==========2. 将任务残余量缩减至设备数量==========")
    timeLines = TimeLineUtil.TimeLines(Device,B,models,BestDecision)
    timeLines.visitAll()
    timeLines.decline()
    timeLines.visitAll()
    # 3. 找到需要划分的任务和执行时间节点
    print("==========3. 找到需要划分的任务和执行时间节点==========")
    deviceList, modelList = timeLines.findRemainModels()
    print(deviceList)
    for i in range(modelList.__len__()):
        print(modelList[i].computeTime)
    # 4. 找到执行时间节点对应剩下的模型
    print("==========4. 找到执行时间节点对应剩下的模型==========")
    for i in range(modelList.__len__()):
        model: MyModel.MyModel = modelList[i]
        print("模型原长度",model.nodes.__len__())
        nodeIndex, computeTime = model.computePartModelWithTime(deviceList[i], Device, B)
        timeline: TimeLineUtil.TimeLine = timeLines.TimeLines[deviceList[i]]
        if timeline.finishTimes.__len__() <= 1:
            timeline.lastTime = 0
        else:
            timeline.lastTime = timeline.finishTimes[timeline.finishTimes.__len__() - 2]
        timeline.lastTime += computeTime
        timeline.finishTimes.pop(timeline.finishTimes.__len__() - 1)
        nodeSize = model.nodes.__len__()
        print(i, nodeSize, nodeIndex)
        # for j in range(model.nodes.__len__()):
        #     print(i, model.nodes[j].id, "修改前",model.nodes[j] ,model.nodes[j].parent_nodes,model.nodes[j].child_nodes)
        model.fixPartModel(nodeIndex, deviceList[i])

    leftTime = []
    for i in range(timeLines.TimeLines.__len__()):
        leftTime.append(timeLines.TimeLines[i].lastTime)
        print("第",i,"台目前的执行时间",timeLines.TimeLines[i].lastTime)
    minTime = min(leftTime)
    for i in range(leftTime.__len__()):
        leftTime[i] -= minTime
        print("第", i, "台目前的执行时间", leftTime[i])

    # 5. 拼接这些模型
    print("==========5. 拼接这些模型==========")
    for i in range(modelList.__len__()):
        model: MyModel.MyModel = modelList[i]
        print(model.name,"合并前",model.nodes.__len__())
        # for j in range(model.nodes.__len__()):
        #     print(i, j, model.name, "修改后",model.nodes[j] ,model.nodes[j].parent_nodes,model.nodes[j].child_nodes)
        model.nodes = readJson.combine_norm_relu(model.nodes)
        model.nodes = readJson.get_sort_nodes(readJson.combine_conv(model.nodes))
        print(i, "合并后", model.nodes.__len__())
        # for j in range(model.nodes.__len__()):
        #     print(j, "合并后",model.nodes[j] ,model.nodes[j].parent_nodes,model.nodes[j].child_nodes)
        # model.nodes = readJson.combine_branch(model.nodes[0])
        print(model.name, "已完成")

    # 6. 划分这些模型
    print("==========6. 划分这些模型==========")
    nowTime = 0
    for i in range(modelList.__len__()):
        timeList = testMyPartition.run_fix_bp(Device, B, modelList[i], leftTime, nowTime)
        nowTime += timeList[0]

    print(nowTime, "+", minTime, nowTime + minTime)

    # 7. PSO-GA分配这些模型

    # 8. 算出最终时间
    return modelList

if __name__ == '__main__':
    exportModels()



