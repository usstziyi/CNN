import torch
from torch import nn


# 卷积层:input_channels -> output_channels
# 大小:不变
def conv_layer(input_channels, output_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1))

# 稠密块
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, output_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            # 第i个卷积层的输入通道数是:
            # 1. 前i个卷积层的输出通道数+当前层的输入通道数
            # 2. 输入通道数
            # 特征重用 + 渐进式增长
            layer.append(conv_layer(output_channels * i + input_channels, output_channels))
        self.net = nn.Sequential(*layer)
        # conv_block(input_channels, output_channels)
        # conv_block(output_channels + input_channels, output_channels)
        # conv_block(output_channels + output_channels + input_channels, output_channels)
        # conv_block(output_channels + output_channels + output_channels + input_channels, output_channels)
        # ...

    def forward(self, X):
        for blk in self.net:
            # 只改变通道数,不改变大小
            Y = blk(X)
            # dim=1: 连接通道维度上每个块的输入和输出
            # 和下一轮conv_block(output_channels * i + input_channels, output_channels)匹配
            X = torch.cat((X, Y), dim=1)
        return X


# 背景知识回顾
# DenseNet 的核心思想是：每一层都接收前面所有层的输出作为输入（通过通道拼接），因此随着层数增加，通道数会迅速膨胀。
# 为了缓解这种通道爆炸问题，DenseNet 在每个稠密块之后加入一个 Transition Block，它通常包含：
#   一个 1×1 卷积（用于压缩通道数）
#   一个 2×2 平均池化(用于下采样，减小空间尺寸)

# 过渡层:只改变通道数,不改变大小
def transition_block(input_channels, output_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        # 压缩通道数
        nn.Conv2d(input_channels, output_channels, kernel_size=1),
        # 下采样:空间尺寸减半
        nn.AvgPool2d(kernel_size=2, stride=2))



class DenseNet(nn.Module):
    def __init__(self, output_channels=64, growth_rate=32):
        super(DenseNet, self).__init__()
        # 初始卷积层
        b1 = nn.Sequential(
            nn.Conv2d(1, output_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(output_channels), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        # 每个稠密块包含4个卷积层
        num_convs_in_dense_blocks = [4, 4, 4, 4]
        blks = []
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            blks.append(DenseBlock(num_convs, output_channels, growth_rate))
            # 上一个稠密块的输出通道数:
            # 1. 前i个卷积层的输出通道数+当前层的输入通道数
            # 2. 输入通道数
            output_channels += num_convs * growth_rate
            # 在稠密块之间添加一个转换层，使通道数量减半
            # 降低特征图的通道数和空间尺寸
            # 在前3个稠密块之后添加转换层，第4个稠密块不添加
            # 为什么不在最后一个稠密块后加转换层？
            # 因为最后要接全局平均池化（AdaptiveAvgPool2d）和分类头，不需要再下采样或压缩通道了。
            # 而且通常最后一个稠密块后的特征直接用于分类。
            if i != len(num_convs_in_dense_blocks) - 1:
                blks.append(transition_block(output_channels, output_channels // 2))
                output_channels = output_channels // 2

        self.net = nn.Sequential(b1, *blks,
                             nn.BatchNorm2d(output_channels), nn.ReLU(),
                             # 全局平均池化:将每个特征图的空间尺寸压缩为1x1
                             nn.AdaptiveAvgPool2d((1, 1)),
                             nn.Flatten(), 
                             nn.Linear(output_channels, 10))

    def forward(self, X):
        return self.net(X)



        


