from net import VGG_11
from torch import nn
import torch
from d2l import torch as d2l
from common import try_gpu, display_model


def train_vgg(model, train_iter, device, num_epochs):
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
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
 


def evaluate_vgg(model, test_iter, device):
    # 评估模型
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
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    # 初始化模型: VGG-11
    # 每个VGG块(卷积层数, 输出通道数)
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    model = VGG_11(conv_arch).to(device)


    # 打印模型参数数量
    display_model(model)

    # 训练模型
    train_vgg(model, train_iter, device, num_epochs)
    # 评估模型
    # evaluate_vgg(model, test_iter, device)

    #  # 可视化预测结果
    # sample_image, sample_label = next(iter(test_iter))
    # sample_image, sample_label = sample_image.to(device), sample_label.to(device)
    # model.eval()
    # with torch.no_grad():
    #     output = model(sample_image)
    #     _, predicted = torch.max(output, 1)
    # d2l.show_images([sample_image.cpu().squeeze(0)], titles=[f"Predicted: {predicted.item()}"])

if __name__ == "__main__":
    main()