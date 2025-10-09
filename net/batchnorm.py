from torch import nn
import torch

# 批量规范化:
# 在每次训练迭代中，我们首先规范化输入，即减去批次的均值，然后除以批次的标准差。
# 这确保了每个特征在每个批次中都有零均值和单位方差，从而加速了训练过程。

class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('moving_mean', torch.zeros(num_features))
        self.register_buffer('moving_var', torch.ones(num_features))
    
    def forward(self, x):
        if self.training:
            # 计算当前批次的均值和方差
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), keepdim=True)
            # 更新移动平均均值和方差
            self.moving_mean = (1 - self.momentum) * self.moving_mean + self.momentum * mean.squeeze()
            self.moving_var = (1 - self.momentum) * self.moving_var + self.momentum * var.squeeze()
        else:
            # 使用移动平均均值和方差
            mean = self.moving_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            var = self.moving_var.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # 归一化
        x = (x - mean) / torch.sqrt(var + self.eps)
        # 缩放和偏移
        x = self.gamma * x + self.beta
        return x
