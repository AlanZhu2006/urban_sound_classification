"""
推理模块
包含 TTA、集成推理、后处理等功能
"""
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from .config import CONFIG, SAVE_DIR, NPY_DIR, TEST_CSV_PATH, RES_WEIGHTS, CLASS_WEIGHTS
from .dataset import UrbanSoundDataset, create_test_loader
from .models import create_model


def predict_with_tta(
    model: torch.nn.Module,
    imgs: torch.Tensor,
    tta_shifts: List[int] = None,
    device: str = None
) -> torch.Tensor:
    """
    Test-Time Augmentation 推理
    
    Args:
        model: 模型
        imgs: 输入图像 [B, C, H, W]
        tta_shifts: 时间偏移列表
        device: 设备
    
    Returns:
        平均预测概率 [B, num_classes]
    """
    if tta_shifts is None:
        tta_shifts = CONFIG["TTA_SHIFTS"]
    if device is None:
        device = CONFIG["DEVICE"]
    
    outputs = []
    
    # 基础预测 (权重更高)
    with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu"):
        base_out = torch.softmax(model(imgs), dim=1)
        outputs.append((base_out, 1.5))
    
    # 时间偏移 TTA
    for shift in tta_shifts:
        if shift == 0:
            continue
        aug = torch.roll(imgs, shifts=shift, dims=3)
        with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu"):
            outputs.append((torch.softmax(model(aug), dim=1), 1.1))
    
    # 频率偏移 TTA
    for shift in [-2, 2]:
        aug = torch.roll(imgs, shifts=shift, dims=2)
        with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu"):
            outputs.append((torch.softmax(model(aug), dim=1), 0.9))
    
    # 增益 TTA
    for gain in [0.9, 1.1]:
        aug = imgs * gain
        with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu"):
            outputs.append((torch.softmax(model(aug), dim=1), 1.0))
    
    # 加权平均
    final_prob = torch.zeros_like(outputs[0][0])
    total_weight = 0
    
    for prob, weight in outputs:
        final_prob += prob * weight
        total_weight += weight
    
    return final_prob / total_weight


def inference_single_config(
    model_type: str,
    res: int,
    prefix: str = "",
    test_df: pd.DataFrame = None,
    use_tta: bool = True
) -> torch.Tensor:
    """
    单个配置的推理
    
    Args:
        model_type: 模型类型
        res: 分辨率
        prefix: 文件名前缀
        test_df: 测试集 DataFrame
        use_tta: 是否使用 TTA
    
    Returns:
        预测概率 [N, num_classes]
    """
    if test_df is None:
        test_df = pd.read_csv(TEST_CSV_PATH)
    
    model_key = f"{model_type}_{res}{prefix}"
    device = CONFIG["DEVICE"]
    
    # 创建数据加载器
    loader = create_test_loader(test_df, NPY_DIR, res)
    
    final_probs = torch.zeros((len(test_df), 10), device=device)
    fold_count = 0
    
    for fold in range(1, 9):
        path = SAVE_DIR / f"{model_key}_fold_{fold}.pth"
        
        if not path.exists():
            continue
        
        # 加载模型
        model = create_model(model_type, input_mels=res)
        
        try:
            state_dict = torch.load(path, map_location=device)
            new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
        except Exception:
            print(f"⚠️ 加载失败: {path.name}")
            continue
        
        model = model.to(device)
        model.eval()
        
        fold_probs = []
        
        with torch.no_grad():
            for imgs, _, _ in tqdm(loader, desc=f"{model_key} F{fold}", leave=False):
                imgs = imgs.to(device)
                
                if use_tta:
                    probs = predict_with_tta(model, imgs)
                else:
                    with torch.amp.autocast(device_type="cuda"):
                        probs = torch.softmax(model(imgs), dim=1)
                
                fold_probs.append(probs)
        
        final_probs += torch.cat(fold_probs)
        fold_count += 1
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    if fold_count > 0:
        final_probs /= fold_count
    
    return final_probs


def ensemble_inference(
    prefix: str = "",
    resolutions: List[int] = None,
    model_types: List[str] = None,
    weights: Dict[str, float] = None,
    use_tta: bool = True
) -> torch.Tensor:
    """
    多模型、多分辨率集成推理
    
    Args:
        prefix: 文件名前缀
        resolutions: 分辨率列表
        model_types: 模型类型列表
        weights: 权重字典 {model_type_res: weight}
        use_tta: 是否使用 TTA
    
    Returns:
        集成预测概率 [N, num_classes]
    """
    if resolutions is None:
        resolutions = CONFIG["FINAL_RESOLUTIONS"]
    if model_types is None:
        model_types = CONFIG["MODELS"]
    
    test_df = pd.read_csv(TEST_CSV_PATH)
    device = CONFIG["DEVICE"]
    
    final_probs = torch.zeros((len(test_df), 10), device=device)
    total_weight = 0.0
    
    print(f"\n🔮 开始集成推理...")
    
    for model_type in model_types:
        for res in resolutions:
            key = f"{model_type}_{res}"
            
            # 获取权重
            if weights:
                w = weights.get(key, 1.0)
            else:
                w = RES_WEIGHTS.get(res, 1.0)
                if model_type == "CRNN":
                    w *= 1.15
            
            print(f"    {key}: weight={w:.2f}")
            
            probs = inference_single_config(
                model_type, res, prefix, test_df, use_tta
            )
            
            final_probs += probs * w
            total_weight += w
    
    final_probs /= total_weight
    
    return final_probs


def apply_class_weights(
    probs: torch.Tensor,
    weights: List[float] = None
) -> torch.Tensor:
    """
    应用类别权重
    
    Args:
        probs: 预测概率 [N, num_classes]
        weights: 类别权重列表
    
    Returns:
        加权后的概率
    """
    if weights is None:
        weights = CLASS_WEIGHTS
    
    device = probs.device
    weight_tensor = torch.tensor(weights, device=device)
    
    return probs * weight_tensor


def apply_power_sharpening(
    probs: torch.Tensor,
    power: float = 1.05
) -> torch.Tensor:
    """
    概率锐化 (增强高置信度预测)
    
    Args:
        probs: 预测概率
        power: 幂次
    
    Returns:
        锐化后的概率
    """
    sharpened = probs ** power
    # 归一化
    sharpened = sharpened / sharpened.sum(dim=1, keepdim=True)
    return sharpened


def apply_confusion_corrections(
    probs: torch.Tensor
) -> torch.Tensor:
    """
    基于混淆矩阵的后处理修正
    
    处理常见的混淆类别对:
    - Engine (5) vs Air_conditioner (0)
    - Street_music (9) vs Children (2) / Dog (3)
    - Jackhammer (7) vs Drilling (4)
    
    Args:
        probs: 预测概率
    
    Returns:
        修正后的概率
    """
    probs = probs.clone()
    
    # Engine (5) vs Air_conditioner (0)
    conflict_mask = (torch.argmax(probs, dim=1) == 5)
    condition = probs[:, 0] > (probs[:, 5] * 0.60)
    probs[conflict_mask & condition, 0] *= 1.5
    
    # Street_music (9) vs Children (2)
    conflict_mask = (torch.argmax(probs, dim=1) == 9)
    condition = probs[:, 2] > (probs[:, 9] * 0.70)
    probs[conflict_mask & condition, 2] *= 1.5
    
    # Street_music (9) vs Dog (3)
    condition = probs[:, 3] > (probs[:, 9] * 0.70)
    probs[conflict_mask & condition, 3] *= 1.5
    
    # Jackhammer (7) vs Drilling (4)
    conflict_mask = (torch.argmax(probs, dim=1) == 7)
    condition = probs[:, 4] > (probs[:, 7] * 0.75)
    probs[conflict_mask & condition, 4] *= 1.4
    
    return probs


def generate_submission(
    probs: torch.Tensor,
    output_path: Path = None,
    apply_postprocess: bool = True
) -> pd.DataFrame:
    """
    生成提交文件
    
    Args:
        probs: 预测概率
        output_path: 输出路径
        apply_postprocess: 是否应用后处理
    
    Returns:
        提交 DataFrame
    """
    if output_path is None:
        output_path = Path("submission.csv")
    
    if apply_postprocess:
        # 应用后处理
        probs = apply_power_sharpening(probs, power=1.05)
        probs = apply_class_weights(probs)
        probs = apply_confusion_corrections(probs)
    
    # 生成预测
    predictions = torch.argmax(probs, dim=1).cpu().numpy()
    
    # 创建提交文件
    test_df = pd.read_csv(TEST_CSV_PATH)
    submission = pd.DataFrame({
        'ID': range(len(test_df)),
        'TARGET': predictions
    })
    
    submission.to_csv(output_path, index=False)
    print(f"✅ 提交文件已保存: {output_path}")
    
    return submission


def hill_climbing_optimize() -> Dict[str, float]:
    """
    基于启发式的权重优化
    
    Returns:
        优化后的权重字典
    """
    print("\n🧗 运行 Hill Climbing 权重优化...")
    
    weights = {}
    
    for model_type in CONFIG["MODELS"]:
        for res in CONFIG["FINAL_RESOLUTIONS"]:
            w = 1.0
            
            # 模型类型权重
            if model_type == "CRNN":
                w *= 1.15
            
            # 分辨率权重
            if res == 128:
                w *= 1.08
            elif res == 64:
                w *= 0.95
            
            weights[f"{model_type}_{res}"] = w
    
    print(f"    推断权重: {weights}")
    
    return weights
