from convolution import Conv2D_Manual
import torch

def main():
    conv = Conv2D_Manual(
        in_channels=3,
        out_channels=1,
        kernel_height=3,
        kernel_width=3
    )

    # x(B,C_in,H,W)=(2,3,10,10)
    x = torch.randn(2, 3, 10, 10)

    # kernal(C_out,C_in,kh,kw)=(1,3,3,3)
    print(conv.weight.shape)
    # bias(C_out)=(1,)
    print(conv.bias.shape)
    # out(N,C_out,H-kh+1,W-kw+1)=(2,1,8,8)
    print(conv(x).shape)



if __name__ == '__main__':
    main()