from On_demand_Fine_grained_Partitioning.MultiTask.MyModel import MyModel

class TimeLines:
    def __init__(self, Devices: list, B: list, models: list, decision: list):
        self.Devices = Devices
        self.B = B
        self.DeviceNum = Devices.__len__()
        self.TimeLines = []
        self.models = models
        self.decision = decision
        for i in range(self.DeviceNum):
            self.TimeLines.append(TimeLine(i))
        for i in range(self.decision.__len__()):
            timeLine: TimeLine = self.TimeLines[decision[i]]
            timeLine.addTimeLine(models[i],self.Devices,self.B)
            timeLine.addTaskLine(models[i],self.Devices,self.B)
    def visitAll(self):
        for i in range(self.DeviceNum):
            timeLine:TimeLine = self.TimeLines[i]
            print(i, timeLine.finishTimes)
            print('deviceFLOPS',timeLine.deviceFLOPS)
            print('deviceRecv', timeLine.deviceRecv)
            print('deviceSend', timeLine.deviceSend)
            print('deviceSendData', timeLine.deviceSendData)


    def decline(self):
        minDevice = self.findMin()
        minTime = self.TimeLines[minDevice].lastTime
        tasksNum = 0
        maxTasksDevice = 0
        maxTasksNum = 0
        # 找到最小时间后未完成任务数量
        for i in range(self.DeviceNum):
            timeLine: TimeLine = self.TimeLines[i]
            taskNum = self.findRemainTask(minTime, timeLine)
            tasksNum += taskNum
            # 找到最大未完成任务数量的设备
            if taskNum > maxTasksNum:
                maxTasksDevice = i
                maxTasksNum = taskNum
        # 如果未完成任务数量<=设备数量，就退出递归
        if tasksNum < self.DeviceNum:
            print("结束")
            return
        print("重新分配")
        # 如果未完成任务数量>设备数量，开始调整任务，将任务数量最多的设备给时间最少的设备
        self.TimeLines[minDevice].addTimeLine(self.TimeLines[maxTasksDevice].removeTimeLine(),self.Devices,self.B)
        self.decline()

    def findRemainTask(self, minTime, timeLine):
        tasksNum = 0
        j = timeLine.models.__len__() - 1
        while j >= 0 and timeLine.finishTimes[j] > minTime:
            tasksNum += 1
            j -= 1
        return tasksNum


    def findMin(self):
        index = 0
        for i in range(self.DeviceNum):
            if self.TimeLines[i].lastTime < self.TimeLines[index].lastTime:
                index = i
        return index

    def findRemainModels(self):
        minDevice = self.findMin()
        minTime = self.TimeLines[minDevice].lastTime
        deviceList = []
        modelList = []
        for i in range(self.DeviceNum):
            timeLine: TimeLine = self.TimeLines[i]
            j = timeLine.models.__len__() - 1
            timeLine: TimeLine = self.TimeLines[i]
            while j >= 0 and timeLine.finishTimes[j] > minTime:
                deviceList.append(i)
                if j <= 1:
                    timeLine.models[j].computeTime = minTime
                else:
                    timeLine.models[j].computeTime = max(0, minTime - timeLine.finishTimes[j-1])
                modelList.append(timeLine.models[j])
                j -= 1
        return deviceList, modelList

    def findMaxModels(self):
        minDevice = self.findMin()
        minTime = self.TimeLines[minDevice].lastTime
        deviceList = []
        modelList = []
        timeList = []
        for i in range(self.DeviceNum):
            timeLine: TimeLine = self.TimeLines[i]
            j = timeLine.models.__len__() - 1
            timeLine: TimeLine = self.TimeLines[i]
            while j >= 0 and timeLine.finishTimes[j] > minTime:
                deviceList.append(i)
                modelList.append(timeLine.models[j])
                timeList.append(timeLine.finishTimes[j])
                j -= 1
        return minTime, deviceList, modelList, timeList

    def findDiffTime(self):
        maxTime = 0
        resTime = 0
        resFLOPS = 0
        for i in range(self.DeviceNum):
            timeLine:TimeLine = self.TimeLines[i]
            maxTime = max(maxTime, timeLine.lastTime)
        for i in range(self.DeviceNum):
            timeLine: TimeLine = self.TimeLines[i]
            timeTemp = maxTime - timeLine.lastTime
            resTime += timeTemp
            resFLOPS += timeTemp * self.Devices[i].p

        return resTime, resFLOPS

class TimeLine:
    def __init__(self, index):
        self.index = index
        self.models = []
        self.finishTimes = []
        self.lastTime = 0
        self.deviceFLOPS = []
        self.deviceRecv = []
        self.deviceSend = []
        self.deviceSendData = []

    def addTimeLine(self, model: MyModel, Device, B):
        self.models.append(model)
        model.startTime = self.lastTime
        #添加时间
        totalTime = 0
        totalTime += model.FLOPS / Device[self.index].p / 1000 / 1000
        if model.device != self.index:
            deviceType = Device[self.index].type
            primDeviceType = Device[model.device].type
            totalTime += model.height_in * model.height_in *model.c_in / B[deviceType][primDeviceType] * 4 / 8 / 1024 / 1024 * 1000
        self.lastTime += totalTime
        self.finishTimes.append(self.lastTime)

    def addTaskLine(self, model: MyModel, Device, B):
        self.deviceFLOPS.append(model.FLOPS)
        self.deviceRecv.append(0)
        self.deviceSend.append([model.device])
        if model.device != self.index:
            deviceType = Device[self.index].type
            primDeviceType = Device[model.device].type
            transTime = model.height_in * model.height_in * model.c_in / B[deviceType][primDeviceType] * 4 / 8 / 1024 / 1024 * 1000
            self.deviceSendData.append([transTime])
        else:
            self.deviceSendData.append([0])

    def removeTaskLine(self):
        i = self.deviceSendData.__len__() - 1
        if i >= 0:
            self.deviceSendData.pop(i)
            self.deviceRecv.pop(i)
            self.deviceSend.pop(i)
            self.deviceFLOPS.pop(i)

    def modifyTaskLine(self, device_FLOPS, device_recv, device_send, device_send_data):
        self.deviceFLOPS.append(device_FLOPS)
        self.deviceRecv.append(device_recv)
        self.deviceSend.append(device_send)
        self.deviceSendData.append(device_send_data)
        print(self.index,"设备的最后任务取消一部分，留下",device_FLOPS,"FLOPS，接下来发给自己")

    def removeTimeLine(self):
        i = self.finishTimes.__len__()
        if i == 1:
            self.finishTimes.pop(0)
            self.lastTime = 0
            return
        lastModel = self.models[i-1]
        self.finishTimes.pop(i-1)
        self.models.pop(i-1)
        self.lastTime = self.finishTimes[i-2]
        return lastModel

