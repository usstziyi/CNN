from torch import nn
import torch

"""
网络结构简要回顾（带 Dropout 位置）：
卷积层 → ReLU → LRN → MaxPooling
卷积层 → ReLU → LRN → MaxPooling
卷积层 → ReLU
卷积层 → ReLU
卷积层 → ReLU → MaxPooling
全连接层(4096 units) → ReLU → Dropout(p=0.5)
全连接层(4096 units) → ReLU → Dropout(p=0.5)
全连接层(1000 units,对应 ImageNet 的 1000 类) → Softmax(无 Dropout)
"""

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征提取层
        self.features = nn.Sequential(
            # (B,1,224,224) -> (B,96,54,54)
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(inplace=True),
            # (B,96,54,54) -> (B,96,26,26)
            nn.MaxPool2d(kernel_size=3, stride=2),

            # (B,96,26,26) -> (B,256,26,26)
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # (B,256,26,26) -> (B,256,12,12)
            nn.MaxPool2d(kernel_size=3, stride=2),

            # (B,256,12,12) -> (B,384,12,12)
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # (B,384,12,12) -> (B,384,12,12)
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # (B,384,12,12) -> (B,256,12,12)
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # (B,256,12,12) -> (B,256,5,5)
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # 展平层
        # (B,256,5,5) -> (B,256*5*5)=(B,6400)
        self.flatten = nn.Flatten()

        # 分类层
        self.classifier = nn.Sequential(
            # 全连接层(6400 units) → ReLU → Dropout(p=0.5)
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            # 全连接层(4096 units) → ReLU → Dropout(p=0.5)
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            # 全连接层(10 units)
            nn.Linear(4096, 10),


            # # Softmax(无 Dropout)
            # nn.Softmax(dim=1)
        )

    # x(B,1,224,224)
    def forward(self, x):
        # 处理后：(B,1,224,224) -> (B,96,54,54) -> (B,96,26,26) -> (B,256,26,26) -> (B,256,12,12) -> (B,384,12,12) -> (B,384,12,12) -> (B,256,12,12) -> (B,256,5,5)
        x = self.features(x)
        # 处理后：(B,256,5,5) -> (B,256*5*5)=(B,6400)
        x = self.flatten(x)
        # 处理后：(B,6400) -> (B,4096) -> (B,4096) -> (B,10)
        x = self.classifier(x)
        return x
