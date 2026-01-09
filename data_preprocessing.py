"""
数据预处理模块 - 数据增强和Transform定义
"""
import torchvision.transforms as transforms


def get_train_transform():
    """获取训练集的数据增强Transform"""
    return transforms.Compose([
        # 合并所有仿射变换为一次操作，大幅减少CPU计算
        transforms.RandomAffine(
            degrees=30,  # 旋转±30度
            translate=(0.05, 0.05),  # 平移
            scale=(0.8, 1.1)  # 缩放
        ),
        # 降低模糊频率，50%概率应用，减轻CPU负担
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 8))
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def get_test_transform():
    """获取测试集的Transform（轻量级增强）"""
    return transforms.Compose([
        # 只保留必要的增强：轻微缩放+模糊（50%概率，降低CPU开销）
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),  # 轻微缩放
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 4))  # 减小kernel和sigma
        ], p=0.5),  # 降低到50%概率
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
