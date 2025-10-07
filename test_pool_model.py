
import torch
from pool import Pool2D_Model


def main():
    # 测试最大池化层
    model = Pool2D_Model(
        mode='max',
        kernel_size=2,
        stride=2,
        padding=0
    )
    x = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
    print(x)
    print(model(x))

    # 测试平均池化层
    model = Pool2D_Model(
        mode='avg',
        kernel_size=2,
        stride=2,
        padding=0
    )
    print(x)
    print(model(x))


    # 测试多通道输入
    x = torch.arange(32, dtype=torch.float32).reshape((1, 2, 4, 4))
    model = Pool2D_Model(
        mode='max',
        kernel_size=2,
        stride=2,
        padding=0
    )
    print(x.shape)
    # 池化层不合并通道，卷积层合并通道
    print(model(x).shape)

if __name__ == '__main__':
    main()
