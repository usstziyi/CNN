from convolution import Conv2D_Manual
import torch

def main():
    conv = Conv2D_Manual(
        out_channels=4,  # K
        in_channels=3,   # C
        kernel_height=3, # Hk
        kernel_width=3   # Wk
    )

    # x(B,C,Hi,Wi)=(2,3,10,10)
    x = torch.randn(2, 3, 10, 10)

    # kernal(K,C,Hk,Wk)=(4,3,3,3)
    print(conv.weight.shape)
    # bias(K)=(4,)
    print(conv.bias.shape)
    # out(B,K,Ho,Wo)=(2,4,8,8)
    print(conv(x).shape)



if __name__ == '__main__':
    main()