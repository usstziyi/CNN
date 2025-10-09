from torch import nn
import torch

# 网络中的网络（NiN）:
# 将每个像素位置视为一个样本，每个通道视为一个特征
# 将所有像素视为一个批次，每个像素的特征向量长度为通道数

def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),nn.ReLU()
    )


class NIN(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征提取层
        self.features = nn.Sequential(
            # (B,1,224,224)->(B,1,54,54)->(B,96,26,26)
            nin_block(1, 96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (B,96,26,26)->(B,256,24,24)->(B,256,12,12)
            nin_block(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (B,256,12,12)->(B,384,12,12)->(B,384,5,5)
            nin_block(256, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (B,384,5,5)
            nn.Dropout(0.5),
            # (B,384,5,5)->(B,10,5,5)
            nin_block(384, 10, kernel_size=3, stride=1, padding=1),
        )
        # 全局平均池化层
        # (B,10,5,5)->(B,10,1,1)
        # 全局平均池化层，将每个通道的 H×W 区域平均池化为一个标量
        # PyTorch 内部会自动计算池化窗口大小和步长，使得输出恰好是你想要的尺寸
        # 比如输入是 5×5，要输出 1×1,它会用一个 5×5 的平均池化窗口（相当于对整个特征图求平均）
        # 所以叫 “Adaptive”（自适应）—— 自动适配输入尺寸，达到指定输出尺寸
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # (B,10,1,1)->(B,10)
        # 把张量从第 1 维开始展平成一维，保留 batch 维
        self.flatten = nn.Flatten()
        # 全连接层，将 10 个特征映射到 10 个类别
        # (B,10)->(B,10)
        self.classifier = nn.Linear(10, 10)


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x