import torch.nn as nn
import torch

class Conv2D_Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        import torch.nn as nn

        # 直接使用内置卷积层
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels,           # C输入通道数
            out_channels=out_channels,         # K输出通道数
            kernel_size=kernel_size,           # S卷积核大小
            stride=1,                          # 步长    
            padding=0,                         # 填充
            bias=True                          # 是否有偏置
        )
       

    def forward(self, x):
        # x(B,C,Hi,Wi)=(2,3,10,10)
        # return(B,K,Ho,Wo)=(2,4,8,8)
        return self.conv_layer(x)
        
