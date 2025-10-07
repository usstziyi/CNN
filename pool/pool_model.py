from torch import nn
import torch

class Pool2D_Model(nn.Module):
    def __init__(self, mode='max', kernel_size=2, stride=2, padding=0):
        super().__init__()
        # 池化层:最大池化
        if mode == 'max':
            self.pool_layer = nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
        # 池化层:平均池化
        elif mode == 'avg':
            self.pool_layer = nn.AvgPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    
    def forward(self, x):
        return self.pool_layer(x)