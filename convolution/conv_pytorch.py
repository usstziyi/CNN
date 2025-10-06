import torch
import torch.nn as nn
import torch.nn.functional as F





class Conv2D_Pytorch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_height, kernel_width):
        super().__init__()
        # 卷积核(K, C, Hk, Wk)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_height, kernel_width))
        # 偏置(K,)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = 1
        self.padding = 0

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)

