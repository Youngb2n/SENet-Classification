import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
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

class VGG(nn.Module):
    def __init__(self, vgg_name,num_classes=2, seon = False):
        super(VGG,self).__init__()
        self.vgg_name = vgg_name
        self.seon = seon
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self,x):
        out = self.features(x)
        out = out.view(out.size(0),-1)
        out = self.classifier(out)
        return F.sigmoid(out)
    
    def _make_layers(self, cfg):
        layers=  []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else :
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=x ,kernel_size=(3,3),stride =1, padding=1,bias=False),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)  # inplace 메모리 감소
                           ]
                if self.seon:
                    layers +=[SELayer(x)]

                in_channels = x
        return nn.Sequential(*layers)


def VGG11(seon = False):
    return VGG('VGG11',seon=seon)
def VGG13(seon = False):
    return VGG('VGG13',seon=seon)
def VGG16(seon = False):
    return VGG('VGG16',seon=seon)
def VGG19(seon = False):
    return VGG('VGG19',seon=seon)    
