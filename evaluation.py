"""
评估分析模块 - 模型性能评估和分析
"""
import torch
import numpy as np
from tqdm import tqdm


def top_k_accuracy(model, loader, device, k_values=[1, 5, 10]):
    """计算Top-K准确率"""
    model.eval()
    correct = {k: 0 for k in k_values}
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='计算Top-K准确率'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # 获取Top-K预测
            _, pred = outputs.topk(max(k_values), dim=1, largest=True, sorted=True)
            
            # 统计准确率
            for k in k_values:
                correct[k] += pred[:, :k].eq(labels.view(-1, 1).expand_as(pred[:, :k])).sum().item()
            
            total += labels.size(0)
    
    # 计算百分比
    acc = {k: 100 * correct[k] / total for k in k_values}
    return acc


def analyze_difficult_classes(model, loader, device, num_classes, dataset, top_n=10):
    """分析最容易混淆的类别
    
    Args:
        model: 训练好的模型
        loader: 数据加载器
        device: 设备
        num_classes: 类别总数
        dataset: 数据集对象（用于获取字符映射）
        top_n: 返回最难/最易的前N个类别
        
    Returns:
        class_accuracy: 每个类别的准确率数组
        difficult_classes: 最难识别的类别索引
        easy_classes: 最易识别的类别索引
    """
    model.eval()
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='分析困难类别'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for label, pred in zip(labels, predicted):
                label_idx = label.item()
                class_total[label_idx] += 1
                if label_idx == pred.item():
                    class_correct[label_idx] += 1
    
    # 计算每个类别的准确率
    class_accuracy = np.divide(class_correct, class_total, 
                               out=np.zeros_like(class_correct), 
                               where=class_total!=0) * 100
    
    # 找出最难和最易识别的类别
    valid_indices = class_total > 0
    valid_acc = class_accuracy[valid_indices]
    valid_classes = np.where(valid_indices)[0]
    
    # 最难识别的类别
    difficult_indices = np.argsort(valid_acc)[:top_n]
    difficult_classes = valid_classes[difficult_indices]
    
    # 最易识别的类别
    easy_indices = np.argsort(valid_acc)[-top_n:][::-1]
    easy_classes = valid_classes[easy_indices]
    
    print(f'\n【最难识别的{top_n}个类别】')
    for i, cls in enumerate(difficult_classes, 1):
        char = dataset.get_char(cls)
        print(f'  {i}. 字符 "{char}" (类别{cls}): {class_accuracy[cls]:.2f}% ({int(class_correct[cls])}/{int(class_total[cls])})')
    
    print(f'\n【最易识别的{top_n}个类别】')
    for i, cls in enumerate(easy_classes, 1):
        char = dataset.get_char(cls)
        print(f'  {i}. 字符 "{char}" (类别{cls}): {class_accuracy[cls]:.2f}% ({int(class_correct[cls])}/{int(class_total[cls])})')
    
    return class_accuracy, difficult_classes, easy_classes


def save_evaluation_report(model_info_path, final_acc, top_k_acc, history, device):
    """保存性能评估报告到文件
    
    Args:
        model_info_path: 配置文件路径
        final_acc: 最终测试准确率
        top_k_acc: Top-K准确率字典
        history: 训练历史字典
        device: 设备信息
    """
    with open(model_info_path, 'a', encoding='utf-8') as f:
        f.write('\n' + '=' * 70 + '\n')
        f.write('ETL8 日文手写汉字识别 - 性能总结报告\n')
        f.write('=' * 70 + '\n')

        f.write('\n【性能指标】\n')
        f.write(f'  - 测试准确率: {final_acc:.2f}%\n')
        if 5 in top_k_acc:
            f.write(f'  - Top-5 准确率: {top_k_acc[5]:.2f}%\n')
        if 10 in top_k_acc:
            f.write(f'  - Top-10 准确率: {top_k_acc[10]:.2f}%\n')
        f.write(f'  - 训练时间: {sum(history["epoch_time"])/60:.2f}分钟\n')
        f.write(f'  - 设备: {device}\n')

        f.write('\n【下一步建议】\n')
        if final_acc < 85:
            f.write('  ⚠ 准确率较低，建议:\n')
            f.write('    - 增加训练epochs\n')
            f.write('    - 尝试更深的网络架构（如ResNet）\n')
            f.write('    - 增强数据增强策略\n')
        elif final_acc < 95:
            f.write('  ✓ 准确率良好，可以考虑:\n')
            f.write('    - 尝试迁移到CRNN架构（准备长文本识别）\n')
            f.write('    - 使用预训练模型微调\n')
            f.write('    - 添加注意力机制\n')
        else:
            f.write('  ✓ 准确率优秀！可以进行下一步:\n')
            f.write('    - 部署模型进行实际应用\n')
            f.write('    - 扩展到CRNN+CTC处理长文本\n')
            f.write('    - 尝试模型压缩和加速\n')

        f.write('\n' + '=' * 70 + '\n')
        f.write('分析完成！所有结果已保存。\n')
        f.write('=' * 70 + '\n')
