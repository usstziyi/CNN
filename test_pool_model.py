
import torch
from pool import Pool2D_Model


def main():
    model = Pool2D_Model(
        mode='max',
        kernel_size=2,
        stride=2,
        padding=0
    )
    x = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
    print(x)
    print(model(x))

    model = Pool2D_Model(
        mode='avg',
        kernel_size=2,
        stride=2,
        padding=0
    )
    print(x)
    print(model(x))

if __name__ == '__main__':
    main()
