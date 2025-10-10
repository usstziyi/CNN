from turtle import right
from torch import nn
import torch
from d2l import torch as d2l
from net import ResNet_18
from common import try_gpu, display_model


# 训练ResNet_18模型
def train_resnet(model, train_loader, device, num_epochs=10, lr=0.05):
    # 初始化模型权重，使用Xavier均匀分布初始化，确保每个层的权重在训练开始时都有一个合理的初始值，
    # 这有助于加速模型的收敛。
    # sigmoid作为激活函数的缺点：
    # 1. 梯度消失问题：sigmoid函数在输入值很大或很小时，梯度接近于0，这会导致在反向传播过程中，梯度值被显著减少，从而使得模型训练变得困难。
    # 2. 输出不是零均值：sigmoid函数的输出范围是(0, 1)，而不是像ReLU函数那样的(-1, 1)。这意味着输出的均值不是0，这在某些情况下可能会影响模型的训练。
    # 所以这里初始化权重时，使用Xavier均匀分布初始化，确保每个层的权重在训练开始时都有一个合理的初始值，
    # 这有助于加速模型的收敛。
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.9)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels) # 每个batch的平均损失
            optimizer.zero_grad()
            loss.backward() # 每个batch的梯度
            optimizer.step()
            with torch.no_grad():
                running_loss += loss.item()
                print(f"Batch Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        # loss: 每个batch的损失
        # running_loss: 每个epoch的总损失
        # len(train_loader): 每个epoch的batch数量
        # running_loss/len(train_loader): 每个epoch的平均损失

# 评估模型
def evaluate_resnet(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            # outputs(256, 10)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Test Accuracy: {100 * correct / total:.2f}%")


def main():
    # 超参数
    num_epochs = 10
    batch_size = 256
    lr = 0.05

    # 获取可用设备
    device = try_gpu()
    print(f"Using device: {device}")
    
    # 加载数据集
    train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size,resize=96)

    # 创建ResNet_18模型并移动到设备
    model = ResNet_18().to(device)

    # 打印模型参数数量
    display_model(model)

    # 训练模型
    train_resnet(model, train_iter, device, num_epochs, lr)

    # 评估模型
    evaluate_resnet(model, test_iter, device)



    


if __name__ == '__main__':
    main()

