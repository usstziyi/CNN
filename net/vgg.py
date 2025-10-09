import torch
from torch import nn

# VGG块
# 特点：
# 1. 卷积层：使用3x3卷积核，填充1，保持输入输出尺寸相同
# 2. 激活函数：ReLU
# 3. 池化层：2x2最大池化，步幅2，输出尺寸减半
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    # 卷积层
    for _ in range(num_convs):
        # kernel_size=3, padding=1 保持输入输出尺寸相同
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels # 下一层的输入通道数等于这一层的输出通道数
    # 池化层
    # kernel_size=2, stride=2 池化窗口2x2, 步幅2, 输出尺寸减半
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)



"""
网络层数:11层(5个VGG块 + 1个Flatten层 + 3个全连接层)

输出通道数:64, 128, 256, 512, 512

输出尺寸 = 输入尺寸 / (2 ^ 池化次数)
         = 224 / (2^5)
         = 224 / 32
         = 7
"""

class VGG_11(nn.Module):
    def __init__(self, conv_arch):
        super().__init__()
        # 卷积层架构
        self.conv_arch = conv_arch
        # 特征提取层
        self.features = self._make_layers()

        # 展平层
        self.flatten = nn.Flatten()

        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )
    
    # 特征提取层
    def _make_layers(self):
        layers = []
        in_channels = 1  # 初始输入通道数（灰度图像）
        for num_convs, out_channels in self.conv_arch:
            layers.append(vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels  # 更新下一个块的输入通道数
        return nn.Sequential(*layers)

    # 前向传播
    def forward(self, x):
        # 特征提取
        x = self.features(x)
        # 展平
        x = self.flatten(x)
        # 分类
        x = self.classifier(x)
        return x

