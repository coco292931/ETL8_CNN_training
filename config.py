"""
配置模块 - ETL8手写汉字识别项目
包含项目的全局配置参数
"""

# 路径配置
ETL8_PATH = r'C:\Users\tonyp\Downloads\unpack_etlcdb\ETL8G'
OUTPUT_DIR = "./bigHomework"

# 数据配置
MAX_SAMPLES = 10000  # 最大样本加载量。153916个样本, 956个字符类别。0表示全部

# 训练超参数
EPOCHS = 50
BATCH_SIZE = 512
LEARNING_RATE = 0.001

# DataLoader配置
NUM_WORKERS = 0  # Windows系统建议设为0
PIN_MEMORY = True

# 模型版本
MODEL_VERSION = "v1.41"

# 学习率调度器配置
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5
