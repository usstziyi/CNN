from net import AlexNet
import torch
from torch import nn
from d2l import torch as d2l
from common import try_gpu




def train_alexnet(model, train_iter, device, num_epochs=10):
    """训练AlexNet模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.9)
    
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
    batch_size = 256
    num_epochs = 10 

    # 获取可用设备
    device = try_gpu()
    print(f"Using device: {device}")
    
    # 加载数据集,此处使用fashion_mnist数据集模拟ResNet-18的输入尺寸224x224
    train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size,resize=224)
 
    # 创建AlexNet模型并移动到设备
    model = AlexNet().to(device)

    # 训练模型
    train_alexnet(model, train_iter, device, num_epochs)

    # 评估模型
    evaluate_alexnet(model, test_iter, device)
 

if __name__ == '__main__':
    main()
