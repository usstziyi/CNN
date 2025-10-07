import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


def load_resnet18_dataset(batch_size=32, data_dir='./data', download=True):
    """
    下载和加载适合ResNet-18的数据集
    
    参数:
        batch_size: 批量大小
        data_dir: 数据存储目录
        download: 是否下载数据集
        
    返回:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        num_classes: 类别数量
    """
    # 创建数据目录
    os.makedirs(data_dir, exist_ok=True)
    
    # ResNet-18的标准数据预处理
    # 使用ImageNet的均值和标准差进行标准化
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # 训练数据增强
    train_transform = transforms.Compose([
        transforms.Resize(256),  # 先缩放到256x256
        transforms.RandomCrop(224),  # 随机裁剪到224x224
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转换为张量
        normalize  # 标准化
    ])
    
    # 测试数据预处理
    test_transform = transforms.Compose([
        transforms.Resize(256),  # 先缩放到256x256
        transforms.CenterCrop(224),  # 中心裁剪到224x224
        transforms.ToTensor(),  # 转换为张量
        normalize  # 标准化
    ])
    
    # 加载CIFAR-10数据集（作为示例，因为ImageNet太大）
    # CIFAR-10有10个类别，适合演示ResNet-18
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=test_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    num_classes = 10  # CIFAR-10有10个类别
    
    print(f"数据集信息:")
    print(f"  训练样本数: {len(train_dataset)}")
    print(f"  测试样本数: {len(test_dataset)}")
    print(f"  类别数: {num_classes}")
    print(f"  图像尺寸: 224x224x3")
    
    return train_loader, test_loader, num_classes


def load_imagenet_subset(batch_size=32, data_dir='./data', download=True):
    """
    加载ImageNet子集（如果需要更真实的数据集）
    注意：ImageNet数据集很大，需要手动下载
    """
    # 这里可以扩展为加载ImageNet子集
    # 例如使用ImageNet-1k的子集或Tiny ImageNet
    pass


if __name__ == '__main__':
    # 测试数据集加载
    train_loader, test_loader, num_classes = load_resnet18_dataset(batch_size=32)
    
    # 打印一个batch的信息
    for images, labels in train_loader:
        print(f"Batch图像尺寸: {images.shape}")  # [batch_size, 3, 224, 224]
        print(f"Batch标签尺寸: {labels.shape}")  # [batch_size]
        print(f"图像值范围: [{images.min():.3f}, {images.max():.3f}]")
        break