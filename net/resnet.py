from torch import nn
import torch


# 残差块作用:
# 主分支: F(x) = 0
# 旁路径: x
# 合并后: H(x) = F(x) + x = 0 + x = x
# 当主分支F(x) = 0时，残差块就成了一个恒等映射,即H(x) = x


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) # 主路径的批量归一化层
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) # 主路径的批量归一化层

        # 如果输入和输出通道数不同，或者步长不为1，需要使用shortcut连接
        self.shortcut = nn.Sequential()
        # 修正两点:
        # 1.输出大小,通过步长改变
        # 2.输出维度,通过1x1卷积层改变通道数
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels) # shortcut连接的批量归一化层
            )
        # 若 in_channels == out_channels 且 stride == 1，shortcut 保持为空 Sequential，
        # 此时 shortcut(x) 直接返回输入 x，实现恒等映射

        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 残差块的输出加上shortcut连接的输出
        out += self.shortcut(x)
        out = self.relu2(out)
        return out

def resnet_block(input_channels, out_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, out_channels, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return blk


class ResNet_18(nn.Module):
    def __init__(self):
        super(ResNet_18, self).__init__()
        b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))
        self.net = nn.Sequential(b1, b2, b3, b4, b5,
                             nn.AdaptiveAvgPool2d((1, 1)),
                             nn.Flatten(), 
                             nn.Linear(512, 10))
    
    def forward(self, x):
        return self.net(x)

        


