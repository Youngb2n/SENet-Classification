import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class _Bottleneck(nn.Module):
    def __init__(self, inplanes,growth_rate):
        super(_Bottleneck,self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes,4*growth_rate,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate,growth_rate,kernel_size=3,padding=1,bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = torch.cat((x, out), dim=1)
        return out

class _Transition(nn.Module):
    def __init__(self, inplanes,outplanes):
        super(_Transition,self).__init__()
        self.bn = nn.BatchNorm2d(inplanes)
        self.conv = nn.Conv2d(inplanes,outplanes,kernel_size=1,bias=False)
        self.relu =nn.ReLU(inplace=True)
    def forward(self,x):
        out = self.conv(self.relu(self.bn(x)))
        out = F.avg_pool2d(out,2)
        return out
class SELayer(nn.Module):
    def __init__(self, planes, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(planes, planes // reduction, bias=False)
        self.relu =  nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(planes // reduction, planes, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(planes, planes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(planes // reduction, planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.avg_pool(x)
        out = out.view(b, c)
        out = self.fc(out)
        out = out.view(b, c, 1, 1)
        return x * out.expand_as(x)

class DenseNet(nn.Module):
    def __init__(self,num_blocks, growth_rate =32, reduction=0.5, num_classes=2, seon=False):
        super(DenseNet,self).__init__()
        num_planes =64
        self.seon =seon
        self.growth_rate = growth_rate

        self.conv1 = nn.Conv2d(3,num_planes,7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(num_planes)
        self.relu = nn.ReLU(inplace= True)
        self.pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.block1 = self._make_layers(_Bottleneck, num_blocks[0], num_planes)
        num_planes += num_blocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.tran1 = _Transition(num_planes,out_planes)
        self.se1 =SELayer(out_planes)
        num_planes = out_planes 

        self.block2 = self._make_layers(_Bottleneck, num_blocks[1], num_planes)
        num_planes += num_blocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.tran2 = _Transition(num_planes,out_planes)
        self.se2 =SELayer(out_planes)
        num_planes = out_planes
        
        self.block3 = self._make_layers(_Bottleneck, num_blocks[2], num_planes)
        num_planes += num_blocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.tran3 = _Transition(num_planes,out_planes)
        self.se3 =SELayer(out_planes)
        num_planes = out_planes

        self.block4 = self._make_layers(_Bottleneck, num_blocks[3], num_planes)
        num_planes += num_blocks[3]*growth_rate
        self.se4 = SELayer(num_planes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.bn2 = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def forward(self,x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.pool(out)
        out = self.tran1(self.block1(out))
        if self.seon:
            out =self.se1(out)
        out = self.tran2(self.block2(out))
        if self.seon:
            out =self.se2(out)
        out = self.tran3(self.block3(out))
        if self.seon:
            out =self.se3(out)
        out = self.relu(self.bn2(self.block4(out)))
        if self.seon:
            out =self.se4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0),-1)
        out = self.linear(out)

        return out

    def _make_layers(self,block,num_blocks,inplanes):
        layers = []
        for i in range(num_blocks):
            layers.append(block(inplanes,self.growth_rate))
            inplanes += self.growth_rate
        return nn.Sequential(*layers)


def DenseNet121(seon):
    return DenseNet([6,12,24,16], growth_rate=32,seon=seon)

def DenseNet169(seon):
    return DenseNet([6,12,32,32], growth_rate=32,seon=seon)

def DenseNet201(seon):
    return DenseNet([6,12,48,32], growth_rate=32,seon=seon)

def DenseNet161(seon):
    return DenseNet([6,12,36,24], growth_rate=48,seon=seon)
