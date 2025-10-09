from torch import nn
import torch

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征提取层
        self.features = nn.Sequential(
            # 第一个卷积层(输入通道数=1, 输出通道数=6, 卷积核大小=5, 填充=2)
            nn.Conv2d(1, 6, kernel_size=5, padding=2), 
            nn.Sigmoid(),
            # 第一个池化层(池化核大小=2, 步长=2)
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            # 第二个卷积层(输入通道数=6, 输出通道数=16, 卷积核大小=5)
            nn.Conv2d(6, 16, kernel_size=5), 
            nn.Sigmoid(),
            # 第二个池化层(池化核大小=2, 步长=2)
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # 展平层
        self.flatten = nn.Flatten()

        # 分类层
        self.classifier = nn.Sequential(
            # 第一个全连接层(输入特征数=16*5*5, 输出特征数=120)
            nn.Linear(16 * 5 * 5, 120), 
            nn.Sigmoid(),
            # 第二个全连接层(输入特征数=120, 输出特征数=84)
            nn.Linear(120, 84), 
            nn.Sigmoid(),
            # 输出层(输入特征数=84, 输出特征数=10)
            nn.Linear(84, 10)
        )

    def forward(self, x):
        # 特征提取
        x = self.features(x)
        # 展平
        x = self.flatten(x)
        # 分类
        x = self.classifier(x)
        return x



class LeNet_BatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征提取层
        self.features = nn.Sequential(
            # 第一个卷积层(输入通道数=1, 输出通道数=6, 卷积核大小=5, 填充=2)
            nn.Conv2d(1, 6, kernel_size=5, padding=2), 
            nn.BatchNorm2d(6),
            nn.Sigmoid(),
            # 第一个池化层(池化核大小=2, 步长=2)
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            # 第二个卷积层(输入通道数=6, 输出通道数=16, 卷积核大小=5)
            nn.Conv2d(6, 16, kernel_size=5), 
            nn.BatchNorm2d(16),
            nn.Sigmoid(),
            # 第二个池化层(池化核大小=2, 步长=2)
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # 展平层
        self.flatten = nn.Flatten()

        # 分类层
        self.classifier = nn.Sequential(
            # 第一个全连接层(输入特征数=16*5*5, 输出特征数=120)
            nn.Linear(16 * 5 * 5, 120), 
            nn.BatchNorm1d(120),
            nn.Sigmoid(),
            # 第二个全连接层(输入特征数=120, 输出特征数=84)
            nn.Linear(120, 84), 
            nn.BatchNorm1d(84),
            nn.Sigmoid(),
            # 输出层(输入特征数=84, 输出特征数=10)
            nn.Linear(84, 10)
        )

    def forward(self, x):
        # 特征提取
        x = self.features(x)
        # 展平
        x = self.flatten(x)
        # 分类
        x = self.classifier(x)
        return x
