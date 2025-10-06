from convolution import Conv2D_Model
import torch
from torch import nn    


def train_conv(model, x, y, epochs=100):
    # 定义损失函数
    loss = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(x)
        l = loss(output, y)
        l.backward()
        optimizer.step()
        if epoch % 2 == 0:
            print(f'Epoch {epoch}, Loss: {l.item()}')


def main():
    model = Conv2D_Model(
        in_channels=1,       # C
        out_channels=1,      # K
        kernel_size=(1, 2)   # Hk=1, Wk=2
    )

    # torch.zeros 默认创建的是 float32 张量
    x = torch.ones((1,1,6,8))
    y = torch.zeros((1,1,6,7))

    # 制造数据集
    x[:,:,:,2:6] = 0
    y[:,:,:,[1, 5]] = torch.tensor([1, -1]).float()

    # 训练模型
    train_conv(model, x, y, epochs=10000)
    
    # 打印模型参数
    print(model.conv_layer.weight.data)
    print(model.conv_layer.bias.data)


if __name__ == '__main__':
    main()