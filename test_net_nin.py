from net import NIN
from torch import nn
import torch
from common import try_gpu, display_model
from d2l import torch as d2l


# 训练
def train_Nin(model, train_iter, device, learning_rate, num_epochs):
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # 训练模型
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

# 评估
def evaluate_Nin(model, test_iter, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_iter:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


def main():
    # 超参数
    num_epochs = 10
    batch_size = 128
    learning_rate = 0.1

    # 设备
    device = try_gpu()

    # 加载数据
    train_loader, test_loader = d2l.load_data_fashion_mnist(batch_size, resize=224)

    # 模型
    model = NIN().to(device)
    display_model(model)

    # 训练
    train_Nin(model, train_loader, device, learning_rate, num_epochs)

    # 评估
    evaluate_Nin(model, test_loader, device)

    

if __name__ == '__main__':
    main()