"""
训练引擎模块 - 训练和评估函数
"""
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    """训练一个epoch (使用混合精度)"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='训练中 (AMP)')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # 使用autocast进行前向传播
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # 使用scaler进行反向传播和优化器步骤
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{running_loss/(pbar.n+1):.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    return running_loss / len(loader), 100 * correct / total


def evaluate(model, loader, criterion, device):
    """评估模型 (使用混合精度)"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='评估中'):
            images, labels = images.to(device), labels.to(device)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss / len(loader), 100 * correct / total
