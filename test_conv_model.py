from convolution import Conv2D_Model
import torch

def main():
    conv = Conv2D_Model(
        in_channels=3,       # C
        out_channels=4,      # K
        kernel_size=(3, 3)   # Hk=Wk
    )

    # x(B,C,Hi,Wi)=(2,3,10,10)
    x = torch.randn(2, 3, 10, 10)

    # out(B,K,Ho,Wo)=(2,4,8,8)
    print(conv(x).shape)



if __name__ == '__main__':
    main()