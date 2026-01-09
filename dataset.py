"""
数据集模块 - ETL8数据集加载器
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import struct
import os


class ETL8Dataset(Dataset):
    """ETL8日文汉字数据集加载器
    
    ETL8G格式：每个记录8199字节
    - 0-2字节：Serial Sheet Number
    - 2-4字节：JIS Kanji Code  
    - 4-12字节：JIS Typical Reading
    - 12-71字节：各种元数据
    - 71-8199字节：图像数据(8128字节, 128x127, 4-bit深度)
    """
    
    def __init__(self, etl8_path, cache_path, transform=None, max_samples=None):
        """
        Args:
            etl8_path: ETL8文件路径（例如：'./ETL8G/ETL8G_01' 或包含多个文件的目录）
            cache_path: 缓存文件路径
            transform: 图像变换
            max_samples: 最大样本数（用于快速测试）
        """
        self.transform = transform
        
        # 尝试从缓存加载
        if os.path.exists(cache_path):
            print(f'正在从缓存加载数据: {cache_path}')
            try:
                # 修复: 为torch.load添加weights_only=False以允许加载numpy等非权重对象
                cache_data = torch.load(cache_path, weights_only=False)
                self.images = cache_data['images']
                self.labels = cache_data['labels']
                self.char_to_idx = cache_data['char_to_idx']
                self.idx_to_char = cache_data['idx_to_char']
                self.jis_to_char = cache_data['jis_to_char']
                self.num_classes = len(self.char_to_idx)
                
                # 如果缓存中的样本数大于max_samples，则进行截断
                if max_samples and len(self.images) > max_samples:
                    print(f'缓存样本数 ({len(self.images)}) > max_samples ({max_samples})，将截断数据。')
                    self.images = self.images[:max_samples]
                    self.labels = self.labels[:max_samples]
                    # 注意：字符映射关系保持不变，以保证类别索引的稳定性
                
                print(f'从缓存加载完成: {len(self.images)} 个样本, {self.num_classes} 个字符类别')
                return # 加载成功，提前返回
            except Exception as e:
                print(f'加载缓存失败: {e}。将重新解析原始数据。')

        # 如果缓存不存在或加载失败，则解析原始数据
        print('未找到缓存或加载失败，开始解析原始ETL8数据...')
        self.images = []
        self.labels = []
        self.char_to_idx = {}  # JIS Code -> 类别索引
        self.idx_to_char = {}  # 类别索引 -> 实际字符
        self.jis_to_char = {}  # JIS Code -> 实际字符（用于调试）
        
        # 判断是文件还是目录
        if os.path.isdir(etl8_path):
            etl8_files = sorted([os.path.join(etl8_path, f) for f in os.listdir(etl8_path) 
                                if f.startswith('ETL8G')])
        else:
            etl8_files = [etl8_path]
        
        print(f'正在加载 {len(etl8_files)} 个文件')
        
        char_count = 0
        for file_path in etl8_files:
            with open(file_path, 'rb') as f:
                while True:
                    # 读取一条记录（8199字节）
                    record = f.read(8199)
                    if len(record) < 8199:
                        break
                    
                    # 提取JIS Kanji Code（2-4字节，大端序16位）
                    jis_code = struct.unpack('>H', record[2:4])[0]
                    
                    # 解码JIS Code为实际字符（参考原生解码器）
                    try:
                        # 构造ISO-2022-JP格式：ESC $ B + JIS Code + ESC ( B
                        jis_hex = f'1b2442{jis_code:04x}1b2842'
                        char = bytes.fromhex(jis_hex).decode('iso2022_jp')
                    except Exception as e:
                        # 解码失败时使用JIS Code的十六进制表示
                        char = f'U+{jis_code:04X}'
                    
                    # 提取图像数据（60-8188字节 = 8128字节）
                    # 图像尺寸：128x127，4位深度
                    img_bytes = record[60:8188]  # 8128字节（修正：从60开始，不是71）
                    
                    # 使用PIL直接解析4位深度图像（与原生unpack.py保持一致）
                    # Image.frombytes参数: (width, height) = (128, 127)
                    img = Image.frombytes('F', (128, 127), img_bytes, 'bit', 4)
                    img = img.convert('L')  # 转换为8位灰度图
                    # 将0-15映射到0-255（乘以16）
                    img = Image.eval(img, lambda x: x * 16)
                    img_data = np.array(img)
                    
                    # 建立字符到索引的映射
                    if jis_code not in self.char_to_idx:
                        self.char_to_idx[jis_code] = char_count
                        self.idx_to_char[char_count] = char  # 索引 -> 字符
                        self.jis_to_char[jis_code] = char    # JIS -> 字符
                        char_count += 1
                    
                    self.images.append(img_data)
                    self.labels.append(self.char_to_idx[jis_code])
                    
                    if max_samples and len(self.images) >= max_samples:
                        break
            
            if max_samples and len(self.images) >= max_samples:
                break
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        self.num_classes = len(self.char_to_idx)
        
        # 保存到缓存
        print(f'解析完成，正在将数据保存到缓存文件: {cache_path}')
        try:
            cache_data = {
                'images': self.images,
                'labels': self.labels,
                'char_to_idx': self.char_to_idx,
                'idx_to_char': self.idx_to_char,
                'jis_to_char': self.jis_to_char
            }
            torch.save(cache_data, cache_path)
            print('缓存保存成功。')
        except Exception as e:
            print(f'保存缓存失败: {e}')

        print(f'加载完成: {len(self.images)} 个样本, {self.num_classes} 个字符类别')
        print(f'字符示例: {list(self.idx_to_char.values())[:10]}')  # 显示前10个字符
    
    def get_char(self, idx):
        """根据类别索引获取对应的字符"""
        return self.idx_to_char.get(idx, f'Unknown({idx})')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        # 转换为PIL Image并立即resize（避免重复resize）
        img = Image.fromarray(img)
        img = img.resize((128, 128), Image.BILINEAR)  # 使用更快的BILINEAR插值
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
