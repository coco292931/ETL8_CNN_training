"""
推理模块 - OCR字符识别推理
"""
import torch
from PIL import Image


def predict_character(model, img_path, dataset, device, transform):
    """
    对单张图片进行OCR字符识别
    
    Args:
        model: 训练好的模型
        img_path: 图片路径
        dataset: 数据集对象（用于获取字符映射）
        device: 设备
        transform: 图像变换
    
    Returns:
        predicted_char: 预测的字符
        confidence: 置信度
        top_k_results: Top-K预测结果
    """
    model.eval()
    
    # 加载图像
    img = Image.open(img_path).convert('L')  # 转为灰度图
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        
        # Top-5预测
        top_k_probs, top_k_indices = probs.topk(5, dim=1)
        
        # 获取最佳预测
        predicted_idx = top_k_indices[0][0].item()
        confidence = top_k_probs[0][0].item() * 100
        predicted_char = dataset.get_char(predicted_idx)
        
        # 构建Top-K结果
        top_k_results = []
        for i in range(5):
            idx = top_k_indices[0][i].item()
            prob = top_k_probs[0][i].item() * 100
            char = dataset.get_char(idx)
            top_k_results.append((char, prob, idx))
    
    return predicted_char, confidence, top_k_results


def batch_ocr_predict(model, image_list, dataset, device, transform):
    """
    批量OCR预测（模拟识别一段文本）
    
    Args:
        model: 训练好的模型
        image_list: 图片路径列表
        dataset: 数据集对象
        device: 设备
        transform: 图像变换
    
    Returns:
        recognized_text: 识别出的文本字符串
        char_confidences: 每个字符的置信度列表
    """
    model.eval()
    recognized_chars = []
    confidences = []
    
    for img_path in image_list:
        char, conf, _ = predict_character(model, img_path, dataset, device, transform)
        recognized_chars.append(char)
        confidences.append(conf)
    
    recognized_text = ''.join(recognized_chars)
    return recognized_text, confidences
