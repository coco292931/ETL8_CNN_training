"""
可视化工具模块 - 数据和结果可视化
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def show_samples(dataset, num_samples=16):
    """显示数据集样本"""
    _, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(num_samples):
        img, label = dataset[i]
        
        # 反标准化
        img = img * 0.5 + 0.5
        img = img.squeeze().numpy()
        
        # 获取实际字符
        char = dataset.dataset.get_char(label) if hasattr(dataset, 'dataset') else dataset.get_char(label)
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(char, fontsize=14)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_training_history(history, output_path):
    """绘制训练历史曲线
    
    Args:
        history: 包含训练历史的字典
        output_path: 保存图片的路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 损失曲线
    axes[0].plot(history['train_loss'], label='训练损失', linewidth=2)
    axes[0].plot(history['test_loss'], label='测试损失', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('损失值', fontsize=12)
    axes[0].set_title('损失曲线', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[1].plot(history['train_acc'], label='训练准确率', linewidth=2)
    axes[1].plot(history['test_acc'], label='测试准确率', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('准确率 (%)', fontsize=12)
    axes[1].set_title('准确率曲线', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'训练历史图表已保存为: {output_path}')
    plt.show()


def visualize_predictions(model, dataset, device, num_samples=16, output_path=None):
    """可视化模型预测结果
    
    Args:
        model: 训练好的模型
        dataset: 测试数据集
        device: 设备
        num_samples: 显示样本数
        output_path: 保存图片的路径（可选）
    """
    model.eval()
    
    _, axes = plt.subplots(4, 4, figsize=(14, 14))
    axes = axes.ravel()
    
    # 随机选择样本
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    correct_count = 0
    
    # 获取原始dataset（处理random_split的情况）
    original_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, label = dataset[idx]
            
            # 预测
            img_input = img.unsqueeze(0).to(device)
            output = model(img_input)
            _, predicted = torch.max(output, 1)
            predicted = predicted.item()
            
            # 获取预测概率
            probs = torch.softmax(output, dim=1)
            confidence = probs[0][predicted].item() * 100
            
            # 反标准化图像
            img_show = img.squeeze() * 0.5 + 0.5
            
            # 获取实际字符
            true_char = original_dataset.get_char(label)
            pred_char = original_dataset.get_char(predicted)
            
            # 显示
            axes[i].imshow(img_show.cpu().numpy(), cmap='gray')
            
            # 判断预测是否正确
            is_correct = (predicted == label)
            if is_correct:
                correct_count += 1
            
            color = 'green' if is_correct else 'red'
            axes[i].set_title(
                f'真实: {true_char}\n预测: {pred_char} ({confidence:.1f}%)',
                color=color,
                fontsize=12,
                fontweight='bold'
            )
            axes[i].axis('off')
    
    plt.suptitle(
        f'预测结果可视化 (准确率: {correct_count}/{num_samples} = {100*correct_count/num_samples:.1f}%)',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'预测结果已保存为: {output_path}')
    
    plt.show()


def plot_topk_accuracy(top_k_acc, output_path=None):
    """可视化Top-K准确率
    
    Args:
        top_k_acc: Top-K准确率字典
        output_path: 保存图片的路径（可选）
    """
    plt.figure(figsize=(10, 6))
    k_list = list(top_k_acc.keys())
    acc_list = list(top_k_acc.values())

    plt.bar(range(len(k_list)), acc_list, color='steelblue', alpha=0.8)
    plt.xticks(range(len(k_list)), [f'Top-{k}' for k in k_list])
    plt.ylabel('准确率 (%)', fontsize=12)
    plt.title('Top-K准确率分析', fontsize=14, fontweight='bold')
    plt.ylim([0, 105])

    for i, acc in enumerate(acc_list):
        plt.text(i, acc + 1, f'{acc:.2f}%', ha='center', fontsize=11, fontweight='bold')

    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Top-K准确率图表已保存为: {output_path}')
    
    plt.show()
