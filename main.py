"""
主程序 - ETL8手写汉字识别
集成所有模块，执行完整的训练和评估流程
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

# 导入自定义模块
from dataset import ETL8Dataset
from data_preprocessing import get_train_transform, get_test_transform
from model import ETL8_CNN
from train import train_epoch, evaluate
from evaluation import top_k_accuracy, analyze_difficult_classes, save_evaluation_report
from visualization import plot_training_history, visualize_predictions, plot_topk_accuracy
from utils import print_inline


# ==================== 重要参数配置 ====================
# 路径配置
ETL8_PATH = r'.\ETL8G'
OUTPUT_DIR = "./ETL8_training_output"
CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')

# 数据配置
MAX_SAMPLES = 10000  # 最大样本加载量（0表示全部，153916个样本）

# 训练超参数
EPOCHS = 10
BATCH_SIZE = 512
LEARNING_RATE = 0.001

# DataLoader配置
NUM_WORKERS = 0  # Windows系统建议设为0,不然有概率卡住训练
PIN_MEMORY = True

# 模型版本
MODEL_VERSION = "v1.41"

# 学习率调度器配置
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5
# ====================================================


def main():
    """主函数 - 执行完整训练流程"""
    
    # ==================== 1. 初始化设置 ====================
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建输出目录
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f'项目输出目录: {OUTPUT_DIR}')
    print(f'缓存目录: {CACHE_DIR}')
    
    # ==================== 2. 数据准备 ====================
    # 缓存文件路径
    cache_tag = f'_{MAX_SAMPLES}' if MAX_SAMPLES else '_full'
    cache_file = os.path.join(CACHE_DIR, f'etl8_cache{cache_tag}.pt')
    print(f'使用缓存文件: {cache_file}')
    
    # 加载数据集
    train_transform = get_train_transform()
    test_transform = get_test_transform()
    
    full_dataset = ETL8Dataset(
        etl8_path=ETL8_PATH,
        cache_path=cache_file,
        transform=train_transform,
        max_samples=MAX_SAMPLES
    )
    
    # 划分训练集和测试集（80% / 20%）
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # 为测试集重新设置transform（去掉数据增强）
    test_dataset.dataset.transform = test_transform
    print(f'训练集大小: {train_size}')
    print(f'测试集大小: {test_size}')
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    print(f'\n批次大小: {BATCH_SIZE}')
    print(f'训练批次数: {len(train_loader)}')
    print(f'测试批次数: {len(test_loader)}')
    
    # ==================== 3. 模型配置 ====================
    # 创建模型
    model = ETL8_CNN(num_classes=full_dataset.num_classes).to(device)
    print(f'\n总参数量: {sum(p.numel() for p in model.parameters()):,}')
    print(f'可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    
    # 模型文件名
    model_params = {
        'version': MODEL_VERSION,
        'samples': MAX_SAMPLES if MAX_SAMPLES else 'full',
        'epochs': EPOCHS,
        'batch': BATCH_SIZE,
        'lr': LEARNING_RATE
    }
    
    model_name = f"etl8_cnn_{model_params['version']}_s{model_params['samples']}_e{model_params['epochs']}_b{model_params['batch']}_lr{model_params['lr']}"
    model_file = os.path.join(OUTPUT_DIR, f"{model_name}.pth")
    model_info = os.path.join(OUTPUT_DIR, f"{model_name}_config.txt")
    
    # 检查是否已存在模型
    if os.path.exists(model_file):
        response = input(f"{model_file}已存在，是否覆盖已有模型？(y/n): ").lower()
        if response != 'y':
            print("未覆盖已有模型。")
            if input("是否加载并继续训练？(y/n): ").lower() == 'y':
                model.load_state_dict(torch.load(model_file))
                print("✓ 已成功加载模型权重，继续训练。")
            else:
                print("未加载已有模型权重，将从头开始训练。")
    
    print(f"\n模型版本: {MODEL_VERSION}")
    print(f"模型文件名: {model_file}")
    print(f"配置文件名: {model_info}")
    
    # 保存配置信息
    config_info = {
        '模型版本': MODEL_VERSION,
        '创建时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        '模型架构': {
            '类型': 'ETL8_CNN',
            '描述': '5层卷积神经网络',
            '输入尺寸': '1×128×128',
            '通道配置': '1→32→64→128→256→256',
            '分类器': 'AdaptiveAvgPool2d + Dropout(0.4) + Linear',
            '防过拟合措施': 'BatchNorm + Dropout(0.4)',
            '参数量': f"{sum(p.numel() for p in model.parameters()):,}"
        },
        '训练配置': {
            '训练样本数': MAX_SAMPLES if MAX_SAMPLES else '全部 153916',
            '训练集大小': train_size,
            '测试集大小': test_size,
            '字符类别数': full_dataset.num_classes,
            'Epochs': EPOCHS,
            'Batch Size': BATCH_SIZE,
            'Learning Rate': LEARNING_RATE,
            '优化器': 'Adam',
            '学习率调度': f'ReduceLROnPlateau(factor={LR_SCHEDULER_FACTOR}, patience={LR_SCHEDULER_PATIENCE})',
            '损失函数': 'CrossEntropyLoss',
            '混合精度': 'AMP enabled'
        },
        '硬件环境': {
            '设备': str(device),
            'GPU型号': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
        }
    }
    
    with open(model_info, 'w', encoding='utf-8') as f:
        f.write(f"{'='*70}\n")
        f.write(f"ETL8 手写汉字识别模型 - 配置信息\n")
        f.write(f"{'='*70}\n\n")
        
        for section, content in config_info.items():
            f.write(f"【{section}】\n")
            if isinstance(content, dict):
                for key, value in content.items():
                    f.write(f"  {key}: {value}\n")
            else:
                f.write(f"  {content}\n")
            f.write("\n")
        
        f.write(f"{'='*70}\n")
    
    print(f"\n✓ 配置文件已保存: {model_info}")
    
    # ==================== 4. 训练准备 ====================
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE
    )
    scaler = GradScaler()
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epoch_time': []
    }
    
    best_acc = 0.0
    
    # ==================== 5. 训练循环 ====================
    print(f'\n开始训练... (共{EPOCHS}个epochs, 使用AMP)')
    print('=' * 70)
    
    start_training_time = time.time()
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        
        # 评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # 学习率调整
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(test_loss)
        new_lr = optimizer.param_groups[0]['lr']
        lr_changed = new_lr < old_lr
        
        epoch_time = time.time() - start_time
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(epoch_time)
        
        # 保存最佳模型
        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
            torch.save(model.state_dict(), model_file)
        
        # 每个epoch打印一行
        status = ''
        if is_best:
            status += '✓最佳 '
        if lr_changed:
            status += f'LR↓{new_lr:.6f} '
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train: L={train_loss:.4f} A={train_acc:.2f}% | "
              f"Test: L={test_loss:.4f} A={test_acc:.2f}% | "
              f"Time={epoch_time:.1f}s | Best={best_acc:.2f}% {status}")
    
    total_time = time.time() - start_training_time
    print(f'\n训练完成! 最佳测试准确率: {best_acc:.2f}%')
    print(f'总训练时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)')
    print('=' * 70)
    
    # ==================== 6. 训练结果可视化 ====================
    history_plot_file = os.path.join(OUTPUT_DIR, f"{model_name}_history.png")
    plot_training_history(history, history_plot_file)
    
    # ==================== 7. 模型评估 ====================
    print('\n' + '=' * 70)
    print('模型性能评估')
    print('=' * 70)
    
    # 加载最佳模型
    model.load_state_dict(torch.load(model_file))
    
    # 最终测试
    final_loss, final_acc = evaluate(model, test_loader, criterion, device)
    
    print(f'\n【最终测试结果】')
    print(f'  测试损失: {final_loss:.4f}')
    print(f'  测试准确率: {final_acc:.2f}%')
    print(f'  错误率: {100-final_acc:.2f}%')
    
    print(f'\n【训练统计】')
    print(f'  总训练时间: {sum(history["epoch_time"]):.2f}秒 ({sum(history["epoch_time"])/60:.2f}分钟)')
    print(f'  平均每epoch时间: {sum(history["epoch_time"])/len(history["epoch_time"]):.2f}秒')
    print(f'  最佳epoch: {history["test_acc"].index(max(history["test_acc"])) + 1}')
    
    print(f'\n【模型信息】')
    print(f'  类别数量: {full_dataset.num_classes}')
    print(f'  参数总量: {sum(p.numel() for p in model.parameters()):,}')
    print(f'  模型大小: {os.path.getsize(model_file) / 1024:.2f} KB')
    
    # Top-K准确率分析
    top_k_acc = top_k_accuracy(model, test_loader, device, k_values=[1, 5, 10, 20])
    
    print('\n【Top-K准确率分析】')
    for k, acc in top_k_acc.items():
        print(f'  Top-{k:2d} 准确率: {acc:.2f}%')
    
    # 可视化Top-K准确率
    topk_file = os.path.join(OUTPUT_DIR, f"{model_name}_topk_accuracy.png")
    plot_topk_accuracy(top_k_acc, topk_file)
    
    # 困难类别分析
    class_acc, difficult_cls, easy_cls = analyze_difficult_classes(
        model, test_loader, device, full_dataset.num_classes, full_dataset, top_n=10
    )
    
    # 预测结果可视化
    prediction_file = os.path.join(OUTPUT_DIR, f"{model_name}_prediction_results.png")
    visualize_predictions(model, test_dataset, device, num_samples=16, output_path=prediction_file)
    
    # 保存评估报告
    save_evaluation_report(model_info, final_acc, top_k_acc, history, device)
    
    print('\n' + '=' * 70)
    print('所有分析完成！结果已保存。')
    print('=' * 70)


if __name__ == '__main__':
    main()
