# ETL8 手写汉字识别项目

基于PyTorch的ETL8日文手写汉字识别系统，使用CNN进行字符分类。

## 项目结构

```
ETL8_CNN_PyTorch_Modules/
├── main.py                # 主程序入口（包含所有重要参数配置）
├── dataset.py             # ETL8数据集加载器
├── data_preprocessing.py  # 数据预处理和Transform
├── model.py               # CNN模型定义
├── train.py               # 训练和评估函数
├── evaluation.py          # 性能评估和分析
├── visualization.py       # 可视化工具
├── inference.py           # 推理接口
├── utils.py               # 辅助工具函数
└── README.md              # 项目说明
```

## 快速开始

### 1. 环境要求

```bash
pip install torch torchvision numpy matplotlib pillow tqdm
```

从[Download links - etlcdb](http://etlcdb.db.aist.go.jp/download-links/)下载ETL8G数据并解压在目录下的ETL8G（或其他，需手动在 main.py 配置）文件夹中。文件夹中正常情况下应该有34个文件。

### 2. 配置参数

在 `main.py` 开头修改重要参数配置：

```python
# ==================== 重要参数配置 ====================
# 路径配置
ETL8_PATH = r'.\ETL8G'
OUTPUT_DIR = "./ETL8_training_output"
CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')

# 数据配置
MAX_SAMPLES = 10000  # 最大样本加载量（0表示全部，153916个样本）

# 训练超参数
EPOCHS = 50
BATCH_SIZE = 512
LEARNING_RATE = 0.001

# DataLoader配置
NUM_WORKERS = 0  # Windows系统建议设为0
PIN_MEMORY = True

# 模型版本
MODEL_VERSION = "v1.00"

# 学习率调度器配置
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5
# ====================================================
```

### 3. 运行训练

```bash
python main.py
```

## 模块说明

### main.py

主程序入口，包含：

- **重要参数配置**：所有关键参数集中在文件开头定义
  - 路径配置（ETL8_PATH, OUTPUT_DIR, CACHE_DIR）
  - 数据配置（MAX_SAMPLES）
  - 训练超参数（EPOCHS, BATCH_SIZE, LEARNING_RATE）
  - DataLoader配置（NUM_WORKERS, PIN_MEMORY）
  - 模型版本和学习率调度器参数
- **完整训练流程**：数据加载、模型训练、性能评估、结果可视化
- **简洁输出**：每个epoch一行，清晰显示训练进度

### dataset.py

ETL8数据集加载器，功能：

- 解析ETL8G二进制格式
- JIS编码转字符映射
- 缓存机制加速加载
- 图像预处理

### data_preprocessing.py

数据增强策略：

- 训练集：旋转、平移、缩放、模糊
- 测试集：轻量级增强

### model.py

CNN模型架构：

- 5层卷积层（32→64→128→256→256）
- BatchNorm + ReLU激活
- 全局平均池化
- Dropout防过拟合

### train.py

训练引擎：

- AMP混合精度训练
- 单epoch训练函数
- 模型评估函数

### evaluation.py

性能评估工具：

- Top-K准确率计算
- 困难类别分析
- 评估报告生成

### visualization.py

可视化工具：

- 样本展示
- 训练曲线绘制
- 预测结果可视化
- Top-K准确率图表

### inference.py

推理接口：

- 单字符识别
- 批量文本识别
- 置信度计算

### utils.py

辅助工具函数：

- `print_inline()` - 行内打印功能（可用于实时更新进度）

## 训练流程

1. **数据准备**：加载ETL8数据集，划分训练/测试集（8:2）
2. **模型构建**：创建CNN模型，配置优化器和学习率调度器
3. **训练循环**：
   - 使用AMP混合精度训练加速
   - 每个epoch显示一行训练进度
   - 自动保存最佳模型
   - 学习率自动调整
4. **性能评估**：Top-K准确率、困难类别分析
5. **结果可视化**：训练曲线、预测结果展示

## 训练输出示例

```
Epoch [1/50] | Train: L=2.1234 A=45.32% | Test: L=1.8912 A=52.45% | Time=45.2s | Best=52.45% ✓最佳
Epoch [2/50] | Train: L=1.8521 A=58.12% | Test: L=1.6321 A=61.23% | Time=44.8s | Best=61.23% ✓最佳
Epoch [3/50] | Train: L=1.6234 A=65.45% | Test: L=1.4512 A=67.89% | Time=45.1s | Best=67.89% ✓最佳
...
```

每个epoch一行输出，包含训练/测试损失和准确率、用时、最佳准确率等信息。

## 输出文件

训练完成后会生成：

- `etl8_cnn_*.pth` - 模型权重文件
- `etl8_cnn_*_config.txt` - 配置和评估报告
- `etl8_cnn_*_history.png` - 训练曲线图
- `etl8_cnn_*_topk_accuracy.png` - Top-K准确率图
- `etl8_cnn_*_prediction_results.png` - 预测结果可视化

## 使用示例

### 训练模型

```python
from main import main
main()
```

### 单独使用推理

```python
from inference import predict_character
from data_preprocessing import get_test_transform

# 加载模型和数据集
# model = ...
# dataset = ...
# device = ...

transform = get_test_transform()
char, confidence, top_k = predict_character(
    model, 
    "test.png", 
    dataset, 
    device, 
    transform
)
print(f"识别结果: {char}, 置信度: {confidence:.2f}%")
```

## 性能指标

- **准确率**：Top-1准确率
- **Top-K准确率**：Top-5, Top-10, Top-20
- **训练时间**：每epoch平均时间
- **模型大小**：参数量和文件大小

## 注意事项

1. **Windows用户**：`NUM_WORKERS = 0` 避免多进程问题
2. **GPU内存**：如果显存不足，降低 `BATCH_SIZE`
3. **数据缓存**：首次运行会生成缓存文件（约需几分钟），后续加载更快
4. **模型保存**：自动保存最佳模型到OUTPUT_DIR目录
5. **训练输出**：每个epoch一行，50个epoch共50行输出

## 配置建议

### 快速测试（小数据集）

```python
MAX_SAMPLES = 1000
EPOCHS = 10
BATCH_SIZE = 128
```

### 完整训练（全部数据）

```python
MAX_SAMPLES = 0  # 使用全部153916个样本
EPOCHS = 50
BATCH_SIZE = 512
```

## 性能指标

- **Top-1准确率**：单次预测的准确率
- **Top-K准确率**：Top-5, Top-10, Top-20准确率
- **训练时间**：每epoch平均时间和总训练时间
- **模型大小**：参数量和模型文件大小
- **类别分析**：最难/最易识别的字符类别

## 后续扩展

- 迁移到CRNN架构处理变长文本
- 添加注意力机制提升性能
- 模型压缩和量化加速
- 部署到Web应用或移动端

## 许可证

MIT License
