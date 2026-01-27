"""
数据增强模块
包含 Mixup, CutMix, SpecAugment, 噪声注入等增强策略
"""
import random
import numpy as np
import torch
from typing import Tuple

from .config import CONFIG


class Augmenter:
    """
    数据增强器类
    
    支持的增强方法:
    - Mixup: 混合两个样本
    - CutMix: 剪切粘贴两个样本
    - SpecAugment: 频谱掩码
    - Gaussian Noise: 高斯噪声
    """
    
    @staticmethod
    def get_batch(
        imgs: torch.Tensor, 
        lbls: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        随机选择增强策略处理批次
        
        Args:
            imgs: 图像张量 [B, C, H, W]
            lbls: 标签张量 [B]
        
        Returns:
            (增强后的图像, 标签A, 标签B, lambda权重)
        """
        r = random.random()
        
        if CONFIG["USE_CUTMIX"] and r < CONFIG["CUTMIX_PROB"]:
            return Augmenter.cutmix(imgs, lbls)
        elif r < CONFIG["CUTMIX_PROB"] + CONFIG["MIXUP_PROB"]:
            return Augmenter.mixup(imgs, lbls)
        
        return imgs, lbls, lbls, 1.0
    
    @staticmethod
    def add_noise(img: torch.Tensor) -> torch.Tensor:
        """
        添加高斯噪声
        
        Args:
            img: 输入图像张量
        
        Returns:
            添加噪声后的图像
        """
        if random.random() < 0.5:
            return img
        
        noise = torch.randn_like(img) * CONFIG["NOISE_LEVEL"]
        return img + noise
    
    @staticmethod
    def mixup(
        x: torch.Tensor, 
        y: torch.Tensor, 
        alpha: float = 0.4
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Mixup 增强
        
        将两个样本按比例混合:
        x_mixed = λ * x1 + (1-λ) * x2
        
        Args:
            x: 输入图像 [B, C, H, W]
            y: 标签 [B]
            alpha: Beta 分布参数
        
        Returns:
            (混合图像, 标签1, 标签2, lambda)
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
        
        batch_size = x.size(0)
        idx = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[idx, :]
        
        return mixed_x, y, y[idx], lam
    
    @staticmethod
    def cutmix(
        x: torch.Tensor, 
        y: torch.Tensor, 
        alpha: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        CutMix 增强
        
        从另一个样本剪切一个矩形区域粘贴到当前样本
        
        Args:
            x: 输入图像 [B, C, H, W]
            y: 标签 [B]
            alpha: Beta 分布参数
        
        Returns:
            (混合图像, 标签1, 标签2, 有效lambda)
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
        
        batch_size = x.size(0)
        idx = torch.randperm(batch_size).to(x.device)
        
        H, W = x.size(2), x.size(3)
        
        # 计算剪切区域大小
        cut_rat = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_rat)
        cut_w = int(W * cut_rat)
        
        # 随机选择剪切中心
        cy = np.random.randint(H)
        cx = np.random.randint(W)
        
        # 计算边界框
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        
        # 执行剪切粘贴
        x[:, :, bby1:bby2, bbx1:bbx2] = x[idx, :, bby1:bby2, bbx1:bbx2]
        
        # 调整 lambda 为实际剪切比例
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
        
        return x, y, y[idx], lam
    
    @staticmethod
    def spec_augment(
        spec_img: torch.Tensor,
        freq_mask_ratio: float = 0.15,
        time_mask_max: int = 30
    ) -> torch.Tensor:
        """
        SpecAugment: 频谱图掩码增强
        
        随机掩盖频率带和时间段
        
        Args:
            spec_img: 输入频谱图 [B, C, F, T]
            freq_mask_ratio: 频率掩码最大比例
            time_mask_max: 时间掩码最大长度
        
        Returns:
            增强后的频谱图
        """
        if random.random() > 0.5:
            return spec_img
        
        augmented = spec_img.clone()
        _, _, F, T = augmented.shape
        
        # 频率掩码
        f_len = random.randint(0, int(F * freq_mask_ratio))
        f0 = random.randint(0, max(1, F - f_len))
        augmented[:, :, f0:f0+f_len, :] = 0.0
        
        # 时间掩码
        t_len = random.randint(0, min(time_mask_max, T // 4))
        t0 = random.randint(0, max(1, T - t_len))
        augmented[:, :, :, t0:t0+t_len] = 0.0
        
        return augmented
    
    @staticmethod
    def time_shift(img: torch.Tensor, shift: int) -> torch.Tensor:
        """
        时间偏移增强
        
        Args:
            img: 输入图像 [B, C, H, W]
            shift: 偏移量 (正数向右, 负数向左)
        
        Returns:
            偏移后的图像
        """
        if shift == 0:
            return img
        
        augmented = torch.zeros_like(img)
        
        if shift > 0:
            augmented[:, :, :, shift:] = img[:, :, :, :-shift]
        else:
            augmented[:, :, :, :shift] = img[:, :, :, -shift:]
        
        return augmented
    
    @staticmethod
    def frequency_shift(img: torch.Tensor, shift: int) -> torch.Tensor:
        """
        频率偏移增强
        
        Args:
            img: 输入图像 [B, C, H, W]
            shift: 偏移量
        
        Returns:
            偏移后的图像
        """
        return torch.roll(img, shifts=shift, dims=2)
    
    @staticmethod
    def gain_augment(img: torch.Tensor, gain: float) -> torch.Tensor:
        """
        增益调整
        
        Args:
            img: 输入图像
            gain: 增益系数
        
        Returns:
            调整后的图像
        """
        return img * gain


def apply_train_augmentation(imgs: torch.Tensor) -> torch.Tensor:
    """
    应用训练时的组合增强
    
    Args:
        imgs: 输入图像批次
    
    Returns:
        增强后的图像
    """
    imgs = Augmenter.add_noise(imgs)
    imgs = Augmenter.spec_augment(imgs)
    return imgs
