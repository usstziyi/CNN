import torch
import torch.nn as nn





class Conv2D_Manual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_height, kernel_width):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_height, kernel_width))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        return corr2d_multi(x, self.weight) + self.bias


# x     (B,  C,  Hi,  Wi)
# kernal(K,  C,  Hk,  Wk)
# out   (B,  K,  Ho,  Wo)
def corr2d_multi(x, kernal):
    # B: 批量大小
    # C: 通道数
    # H: 输入高度
    # W: 输入宽度
    B, C, H, W = x.shape
    # K: 卷积核组数
    # C: 通道数
    # Hk: 卷积核高度    
    # Wk: 卷积核宽度
    K, _, Hk, Wk = kernal.shape 
    Ho = H - Hk + 1
    Wo = W - Wk + 1
    out = torch.zeros(B, K, Ho, Wo, device=x.device)
    # B 批次这个维度可以并行计算，每个样本独立计算，不需要循环
    # 第k组卷积核，产出第k个输出通道
    for k in range(K):
        # kernal_k(C,Hk,Wk)

        # 用第k组卷积核
        kernal_k = kernal[k]
        for i in range(Ho): # 高度
            for j in range(Wo): # 宽度
                # x                       (B,  C,  Hi, Wi)
                # x[:, :, i:i+Hk, j:j+Wk] (B,  C,  Hk, Wk)
                # kernal_k                (    C,  Hk, Wk) -广播-> (1,  C,  Hk, Wk)
                # temp                    (B,  C,  Hk, Wk)
                temp = x[:, :, i:i+Hk, j:j+Wk] * kernal_k
                # C个输入通道 对 C个卷积核通道进行卷积，产生C层特征图
                # sum(dim=(-2,-1)) 对 Hk,Wk 进行求和，得到一个标量
                # sum(dim=(1)) 对 C 个通道进行求和，得到一个标量
                # 最后这个标量就是第k个输出通道在(i,j)位置的特征值
                # temp                    (B,            )
                temp = temp.sum(dim=(-3,-2,-1))
                # out                     (B,  K,  Ho, Wo)
                out[:, k, i, j] = temp

                
    # out(B,K,Ho,Wo)
    # out(所有样本,k个输出通道,Ho,Wo)
    return out