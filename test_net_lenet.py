import torch
from torch import nn
from d2l import torch as d2l
from net.lenet import LeNet


def try_gpu():
    """检测可用设备：GPU、MPS或CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')



# 训练LeNet模型
def train_lenet(model, train_loader, device, num_epochs=10):
    # 初始化模型权重，使用Xavier均匀分布初始化，确保每个层的权重在训练开始时都有一个合理的初始值，
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
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward() # 每个
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
def evaluate_lenet(model, test_loader, device):
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
    # 获取可用设备
    device = try_gpu()
    print(f"Using device: {device}")
    
    # 加载数据集
    batch_size = 256
    train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

    # 创建LeNet模型并移动到设备
    model = LeNet().to(device)

    # 训练模型
    train_lenet(model, train_iter, device, num_epochs=10)

    # 评估模型
    evaluate_lenet(model, test_iter, device)

if __name__ == '__main__':
    main()


