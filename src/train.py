"""
训练模块
包含单折训练、多分辨率训练、伪标签训练等功能
"""
import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from .config import CONFIG, SAVE_DIR, NPY_DIR, TRAIN_CSV_PATH
from .dataset import UrbanSoundDataset, create_data_loaders
from .augmentation import Augmenter
from .models import create_model, FocalLoss
from .utils import seed_everything


def apply_train_augmentation(imgs: torch.Tensor) -> torch.Tensor:
    """
    应用训练时的数据增强
    
    Args:
        imgs: 输入图像 [B, C, H, W]
    
    Returns:
        增强后的图像
    """
    imgs = Augmenter.add_noise(imgs)
    imgs = Augmenter.spec_augment(imgs)
    return imgs


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_path: Path,
    epochs: int = None,
    patience: int = None,
    device: str = None
) -> Tuple[nn.Module, float]:
    """
    训练单个 Fold
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        save_path: 模型保存路径
        epochs: 训练轮数
        patience: 早停耐心值
        device: 设备
    
    Returns:
        (最佳模型, 最佳准确率)
    """
    if epochs is None:
        epochs = CONFIG["EPOCHS"]
    if patience is None:
        patience = CONFIG["PATIENCE"]
    if device is None:
        device = CONFIG["DEVICE"]
    
    model = model.to(device)
    
    # 尝试编译模型 (PyTorch 2.0+)
    try:
        model = torch.compile(model)
    except Exception:
        pass
    
    # 训练组件
    scaler = torch.amp.GradScaler("cuda" if "cuda" in device else "cpu")
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LR"], weight_decay=5e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=CONFIG["MIN_LR"]
    )
    
    best_acc = 0.0
    no_improvement = 0
    
    for epoch in range(epochs):
        # ==================== 训练阶段 ====================
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for imgs, lbls in pbar:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            
            # 数据增强
            imgs = apply_train_augmentation(imgs)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu"):
                # Mixup/CutMix
                imgs, labels_a, labels_b, lam = Augmenter.get_batch(imgs, lbls)
                outputs = model(imgs)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # ==================== 验证阶段 ====================
        # 早期 epoch 跳过验证加速训练
        if epoch < 10 or (epoch % 2 != 0 and epoch < 35):
            continue
        
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                
                with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu"):
                    outputs = model(imgs)
                
                all_preds.extend(outputs.argmax(1).cpu().numpy())
                all_targets.extend(lbls.numpy())
        
        acc = accuracy_score(all_targets, all_preds)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            no_improvement = 0
        else:
            no_improvement += 1
        
        if no_improvement >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break
    
    # 加载最佳模型
    if save_path.exists():
        state_dict = torch.load(save_path, map_location=device)
        new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        try:
            model.load_state_dict(new_state_dict)
        except Exception:
            model.load_state_dict(state_dict)
    
    return model, best_acc


def train_model(
    csv_path: Path = None,
    prefix: str = "",
    res: int = 128,
    model_type: str = "CRNN",
    skip_existing: bool = True
) -> float:
    """
    训练单个模型配置 (8-Fold)
    
    Args:
        csv_path: 训练 CSV 路径
        prefix: 模型文件名前缀
        res: Mel 频谱分辨率
        model_type: 模型类型
        skip_existing: 是否跳过已存在的模型
    
    Returns:
        平均验证准确率
    """
    seed_everything(CONFIG["SEED"])
    
    if csv_path is None:
        csv_path = TRAIN_CSV_PATH
    
    full_df = pd.read_csv(csv_path)
    model_key = f"{model_type}_{res}{prefix}"
    
    print(f"\n{'='*50}")
    print(f"🚀 Training: {model_key}")
    print(f"{'='*50}")
    
    fold_scores = []
    
    for fold_id in range(1, 9):
        save_path = SAVE_DIR / f"{model_key}_fold_{fold_id}.pth"
        
        # 检查是否已存在
        if skip_existing and save_path.exists():
            print(f"    ⏩ Fold {fold_id}: 模型已存在，跳过训练")
            
            # 仍然计算该 fold 的验证分数
            model = create_model(model_type, input_mels=res)
            model.load_state_dict(torch.load(save_path, map_location=CONFIG["DEVICE"]))
            model = model.to(CONFIG["DEVICE"])
            model.eval()
            
            val_df = full_df[full_df['fold'] == fold_id].reset_index(drop=True)
            val_loader = DataLoader(
                UrbanSoundDataset(val_df, NPY_DIR, 'train', res, cache_data=True),
                batch_size=CONFIG["BATCH_SIZE"] * 2,
                shuffle=False,
                num_workers=CONFIG["NUM_WORKERS"],
                pin_memory=True
            )
            
            preds, targets = [], []
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs = imgs.to(CONFIG["DEVICE"])
                    outputs = model(imgs)
                    preds.extend(outputs.argmax(1).cpu().numpy())
                    targets.extend(lbls.numpy())
            
            fold_acc = accuracy_score(targets, preds)
            fold_scores.append(fold_acc)
            print(f"    [Fold {fold_id}] Accuracy: {fold_acc:.4f}")
            
            del model
            gc.collect()
            torch.cuda.empty_cache()
            continue
        
        print(f"    🔥 Fold {fold_id}: 开始训练...")
        
        # 划分数据
        train_df = full_df[full_df['fold'] != fold_id].reset_index(drop=True)
        val_df = full_df[full_df['fold'] == fold_id].reset_index(drop=True)
        
        # 创建数据加载器
        train_loader, val_loader = create_data_loaders(
            train_df, val_df, NPY_DIR, res,
            CONFIG["BATCH_SIZE"], CONFIG["NUM_WORKERS"]
        )
        
        # 创建模型
        model = create_model(model_type, input_mels=res)
        
        # 训练
        model, best_acc = train_one_fold(
            model, train_loader, val_loader, save_path
        )
        
        fold_scores.append(best_acc)
        print(f"    [Fold {fold_id}] Best Accuracy: {best_acc:.4f}")
        
        del model, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()
    
    avg_score = np.mean(fold_scores)
    print(f"\n📊 {model_key} Average: {avg_score:.4f}")
    
    return avg_score


def train_multi_resolution(
    csv_path: Path = None,
    prefix: str = "",
    resolutions: List[int] = None,
    model_types: List[str] = None
) -> Dict[str, float]:
    """
    多分辨率、多模型训练
    
    Args:
        csv_path: 训练 CSV 路径
        prefix: 文件名前缀
        resolutions: 分辨率列表
        model_types: 模型类型列表
    
    Returns:
        各配置的平均分数字典
    """
    if resolutions is None:
        resolutions = CONFIG["FINAL_RESOLUTIONS"]
    if model_types is None:
        model_types = CONFIG["MODELS"]
    
    scores = {}
    
    for model_type in model_types:
        for res in resolutions:
            key = f"{model_type}_{res}{prefix}"
            
            # 动态调整 batch size
            batch_size = CONFIG["BATCH_SIZE"]
            if res >= 320:
                batch_size = 48
            elif res >= 256:
                batch_size = 56
            
            CONFIG["BATCH_SIZE"] = batch_size
            
            score = train_model(csv_path, prefix, res, model_type)
            scores[key] = score
    
    return scores


def create_pseudo_labels(
    probs: torch.Tensor,
    test_df: pd.DataFrame,
    threshold: float = 0.90
) -> Optional[pd.DataFrame]:
    """
    生成伪标签数据
    
    Args:
        probs: 预测概率 [N, num_classes]
        test_df: 测试集 DataFrame
        threshold: 置信度阈值
    
    Returns:
        伪标签 DataFrame 或 None
    """
    max_probs, preds = torch.max(probs, dim=1)
    
    # 筛选高置信度样本
    mask = max_probs > threshold
    pseudo_df = test_df[mask.numpy()].copy()
    
    if len(pseudo_df) < 50:
        print(f"⚠️ 伪标签样本不足 ({len(pseudo_df)}), 跳过")
        return None
    
    print(f"🔥 选中 {len(pseudo_df)} 个伪标签样本 (阈值: {threshold})")
    
    pseudo_df['classID'] = preds[mask].numpy()
    pseudo_df['fold'] = np.random.randint(1, 9, size=len(pseudo_df))
    
    return pseudo_df


def merge_pseudo_labels(
    original_csv: Path,
    pseudo_df: pd.DataFrame,
    output_path: Path
) -> Path:
    """
    合并原始训练集和伪标签
    
    Args:
        original_csv: 原始训练 CSV
        pseudo_df: 伪标签 DataFrame
        output_path: 输出路径
    
    Returns:
        合并后的 CSV 路径
    """
    original_df = pd.read_csv(original_csv)
    merged_df = pd.concat([original_df, pseudo_df]).reset_index(drop=True)
    merged_df.to_csv(output_path, index=False)
    
    print(f"✅ 合并完成: {len(original_df)} + {len(pseudo_df)} = {len(merged_df)}")
    
    return output_path
