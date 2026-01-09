"""
模型定义模块 - ETL8_CNN模型
"""
import torch.nn as nn


class ETL8_CNN(nn.Module):
    """用于ETL8字符识别的卷积神经网络
    
    架构设计思路：
    - 输入: 1×128×128 灰度图
    - 通道递增: 32→64→128→256
    - 前期: 小卷积核堆叠 + 池化降维
    - 后期: 空间降到8×8后只做卷积，保留结构细节
    - 输出: 全局平均池化 + 小型全连接层
    """
    
    def __init__(self, num_classes):
        super(ETL8_CNN, self).__init__()
        
        # 卷积层1: 1 -> 32 (128×128 -> 64×64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # 卷积层2: 32 -> 64 (64×64 -> 32×32)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # 卷积层3: 64 -> 128 (32×32 -> 16×16)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # 卷积层4: 128 -> 256 (16×16 -> 8×8)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # 卷积层5: 256 -> 256 (8×8保持，不池化，保留细节)
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # 全局平均池化 + 分类层
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化: 8×8 -> 1×1
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)  # 细节保留层
        x = self.classifier(x)
        return x
