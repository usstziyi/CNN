import torch
from torch import nn
from common import load_resnet18_dataset, try_gpu


def test_resnet18_dataset():
    """测试ResNet-18数据集加载函数"""
    
    # 获取可用设备
    device = try_gpu()
    print(f"使用设备: {device}")
    
    # 加载ResNet-18数据集
    print("正在加载ResNet-18数据集...")
    train_loader, test_loader, num_classes = load_resnet18_dataset(
        batch_size=32,
        data_dir='./data/cifar10',
        download=True
    )
    
    print(f"\n数据集加载完成!")
    print(f"训练集批次数量: {len(train_loader)}")
    print(f"测试集批次数量: {len(test_loader)}")
    print(f"类别数量: {num_classes}")
    
    # 检查一个batch的数据
    print(f"\n检查第一个batch的数据:")
    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx == 0:
            print(f"图像张量形状: {images.shape}")  # [batch_size, 3, 224, 224]
            print(f"标签张量形状: {labels.shape}")  # [batch_size]
            print(f"图像数据类型: {images.dtype}")
            print(f"标签数据类型: {labels.dtype}")
            print(f"图像值范围: [{images.min():.3f}, {images.max():.3f}]")
            
            # 将数据移动到设备
            images = images.to(device)
            labels = labels.to(device)
            print(f"移动到设备后的图像形状: {images.shape}")
            break
    
    print(f"\n数据集测试完成!")


def create_simple_resnet18_model(num_classes=10):
    """创建一个简化的ResNet-18模型用于测试"""
    # 这里可以使用torchvision的预训练ResNet-18
    from torchvision import models
    
    model = models.resnet18(pretrained=False, num_classes=num_classes)
    return model


def test_model_with_dataset():
    """测试模型与数据集的兼容性"""
    
    device = try_gpu()
    
    # 加载数据集
    train_loader, test_loader, num_classes = load_resnet18_dataset(
        batch_size=16,  # 使用较小的batch_size进行测试
        download=True
    )
    
    # 创建模型
    model = create_simple_resnet18_model(num_classes)
    model = model.to(device)
    
    print(f"模型创建完成，输入尺寸: 224x224x3")
    print(f"模型输出尺寸: {num_classes}个类别")
    
    # 测试前向传播
    model.eval()
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            print(f"模型输出形状: {outputs.shape}")  # [batch_size, num_classes]
            print(f"前向传播测试成功!")
            break


if __name__ == '__main__':
    print("=" * 50)
    print("ResNet-18数据集加载测试")
    print("=" * 50)
    
    # 测试数据集加载
    test_resnet18_dataset()
    
    print("\n" + "=" * 50)
    print("模型与数据集兼容性测试")
    print("=" * 50)
    
    # 测试模型兼容性
    test_model_with_dataset()