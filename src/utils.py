"""
工具函数模块
包含随机种子、模型工具等通用函数
"""
import os
import gc
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional

from .config import CONFIG


def seed_everything(seed: int = None) -> None:
    """
    设置所有随机种子以确保可复现性
    
    Args:
        seed: 随机种子
    """
    if seed is None:
        seed = CONFIG["SEED"]
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🌱 随机种子已设置: {seed}")


def cleanup_memory() -> None:
    """清理 GPU 和 CPU 内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_device() -> str:
    """
    获取可用设备
    
    Returns:
        设备字符串
    """
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🎮 使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("💻 使用 CPU")
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    统计模型参数量
    
    Args:
        model: PyTorch 模型
    
    Returns:
        可训练参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: torch.nn.Module, name: str = "Model") -> None:
    """
    打印模型摘要信息
    
    Args:
        model: PyTorch 模型
        name: 模型名称
    """
    total_params = count_parameters(model)
    print(f"\n📦 {name} 参数统计:")
    print(f"    可训练参数: {total_params:,}")
    print(f"    约 {total_params / 1e6:.2f}M 参数")


def copy_model_weights(
    src_model: torch.nn.Module,
    dst_model: torch.nn.Module,
    strict: bool = False
) -> None:
    """
    复制模型权重
    
    Args:
        src_model: 源模型
        dst_model: 目标模型
        strict: 是否严格匹配
    """
    src_state = src_model.state_dict()
    dst_state = dst_model.state_dict()
    
    copied = 0
    for name, param in src_state.items():
        clean_name = name.replace('_orig_mod.', '')
        if clean_name in dst_state:
            if dst_state[clean_name].shape == param.shape:
                dst_state[clean_name].copy_(param)
                copied += 1
    
    print(f"✅ 复制了 {copied}/{len(src_state)} 个参数")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Path,
    **kwargs
) -> None:
    """
    保存训练检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮次
        loss: 当前损失
        path: 保存路径
        **kwargs: 其他要保存的数据
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    torch.save(checkpoint, path)
    print(f"💾 检查点已保存: {path}")


def load_checkpoint(
    path: Path,
    model: torch.nn.Module = None,
    optimizer: torch.optim.Optimizer = None,
    device: str = None
) -> Dict[str, Any]:
    """
    加载训练检查点
    
    Args:
        path: 检查点路径
        model: 模型 (可选)
        optimizer: 优化器 (可选)
        device: 设备
    
    Returns:
        检查点数据字典
    """
    if device is None:
        device = CONFIG["DEVICE"]
    
    checkpoint = torch.load(path, map_location=device)
    
    if model is not None:
        state_dict = checkpoint['model_state_dict']
        # 处理 torch.compile 的前缀
        new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"📂 检查点已加载: {path}")
    return checkpoint


def format_time(seconds: float) -> str:
    """
    格式化时间
    
    Args:
        seconds: 秒数
    
    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        return f"{seconds / 3600:.1f}h"


class AverageMeter:
    """计算和存储平均值和当前值"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """早停机制"""
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


def print_config() -> None:
    """打印当前配置"""
    print("\n📋 当前配置:")
    for key, value in CONFIG.items():
        print(f"    {key}: {value}")
