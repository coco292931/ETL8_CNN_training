"""
工具函数模块 - 辅助功能
"""
import sys


def print_inline(text):
    """
    在同一行打印内容（覆盖上一次输出）
    用于训练进度显示，每个epoch只占一行
    
    Args:
        text: 要打印的文本
    """
    print(f'\r{text}', end='', flush=True)

