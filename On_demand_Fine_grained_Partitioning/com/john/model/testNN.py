import torch
import torch.nn as nn
import time

class AlexNet(nn.Module):
    def __init__(self,num_classes=1000):
        super(AlexNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1()
        return x

def runTime(net, input):
    startTime = time.time()
    out = net(input)
    endTime = time.time()
    print((endTime - startTime)*1000)
    print(out)

if __name__ == '__main__':
    alex = AlexNet()
    input = torch.randn(1,3,224,224)
    runTime(alex,input)

