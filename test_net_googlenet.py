from net import GoogLeNet
from torch import nn
import torch
from common import display_model,try_gpu
from d2l import torch as d2l

def train_alexnet(model, train_iter, device, lr, num_epochs=10):
    """训练AlexNet模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # 初始化模型参数, 所有的线性层和卷积层使用 Xavier 初始化, 偏置初始化为0
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_iter:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels) # 每个batch的平均损失
            optimizer.zero_grad()
            loss.backward() # 每个batch的梯度
            optimizer.step()
            with torch.no_grad():
                running_loss += loss.item()
                print(f"Batch Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_iter):.4f}")
        # loss: 每个batch的损失
        # running_loss: 每个epoch的总损失
        # len(train_iter): 每个epoch的batch数量
        # running_loss/len(train_iter): 每个epoch的平均损失
 

def evaluate_alexnet(model, test_iter, device):
    """评估AlexNet模型"""
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_iter:
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
    lr = 0.1

    # 设备
    device = try_gpu()

    # 数据集
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=128,resize=224)
    num_classes = 10
    in_channels = 1
    
    # 模型
    model = GoogLeNet(
        in_channels=in_channels, 
        num_classes=num_classes
        ).to(device)
    display_model(model)

    # 训练
    train_alexnet(model, train_iter, device, lr, num_epochs)

    # 评估
    evaluate_alexnet(model, test_iter, device)







if __name__ == '__main__':
    main()


