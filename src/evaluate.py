"""
评估模块
包含模型评估、混淆矩阵可视化、权重建议等
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

from .config import (
    CONFIG, TRAIN_CSV_PATH, NPY_DIR, SAVE_DIR, 
    CLASS_NAMES, CLASS_NAME_MAP
)
from .dataset import UrbanSoundDataset
from .models import create_model


def evaluate_fold(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: str = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    评估单个 fold 的模型
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        device: 设备
    
    Returns:
        (准确率, 真实标签, 预测标签)
    """
    if device is None:
        device = CONFIG["DEVICE"]
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    return accuracy, all_labels, all_preds


def local_validation(
    model_type: str = "CRNN",
    resolutions: List[int] = None,
    prefix: str = ""
) -> Dict[str, float]:
    """
    本地交叉验证评估
    
    Args:
        model_type: 模型类型
        resolutions: 分辨率列表
        prefix: 模型文件前缀
    
    Returns:
        评估结果字典
    """
    if resolutions is None:
        resolutions = CONFIG["FINAL_RESOLUTIONS"]
    
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    device = CONFIG["DEVICE"]
    
    results = {}
    
    for res in resolutions:
        print(f"\n📊 评估 {model_type}_{res}{prefix}...")
        
        fold_accs = []
        all_preds = []
        all_labels = []
        
        for fold in range(1, 9):
            model_key = f"{model_type}_{res}{prefix}"
            path = SAVE_DIR / f"{model_key}_fold_{fold}.pth"
            
            if not path.exists():
                continue
            
            # 加载模型
            model = create_model(model_type, input_mels=res)
            
            try:
                state_dict = torch.load(path, map_location=device)
                new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict)
            except Exception as e:
                print(f"    ⚠️ Fold {fold} 加载失败: {e}")
                continue
            
            model = model.to(device)
            model.eval()
            
            # 准备验证集
            val_df = train_df[train_df['fold'] == fold]
            val_dataset = UrbanSoundDataset(
                val_df, NPY_DIR, 
                augment=False, 
                target_mels=res
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=CONFIG["BATCH_SIZE"],
                shuffle=False,
                num_workers=CONFIG["NUM_WORKERS"]
            )
            
            # 评估
            acc, labels, preds = evaluate_fold(model, val_loader, device)
            fold_accs.append(acc)
            all_preds.extend(preds)
            all_labels.extend(labels)
            
            print(f"    Fold {fold}: {acc:.4f}")
            
            del model
            torch.cuda.empty_cache()
        
        if fold_accs:
            mean_acc = np.mean(fold_accs)
            std_acc = np.std(fold_accs)
            results[f"{model_type}_{res}"] = mean_acc
            print(f"    ✅ 平均准确率: {mean_acc:.4f} ± {std_acc:.4f}")
            
            # 打印分类报告
            if all_labels and all_preds:
                print("\n    分类报告:")
                print(classification_report(
                    all_labels, all_preds,
                    target_names=CLASS_NAMES,
                    digits=4
                ))
    
    return results


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path = None,
    title: str = "Confusion Matrix"
) -> None:
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        save_path: 保存路径
        title: 标题
    """
    if not HAS_PLOTTING:
        print("⚠️ 未安装 matplotlib/seaborn, 无法绘制混淆矩阵")
        return
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✅ 混淆矩阵已保存: {save_path}")
    
    plt.show()


def suggest_class_weights(y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
    """
    基于混淆矩阵分析建议类别权重
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
    
    Returns:
        建议的类别权重列表
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # 计算每个类别的召回率
    recalls = np.diag(cm) / cm.sum(axis=1)
    
    # 召回率低的类别给更高权重
    weights = 1.0 / (recalls + 0.1)  # 加 0.1 防止除零
    
    # 归一化到均值为 1
    weights = weights / weights.mean()
    
    print("\n📊 类别权重建议:")
    for i, (name, recall, weight) in enumerate(zip(CLASS_NAMES, recalls, weights)):
        print(f"    {name}: recall={recall:.3f}, weight={weight:.3f}")
    
    return weights.tolist()


def analyze_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filenames: List[str] = None,
    top_k: int = 10
) -> pd.DataFrame:
    """
    分析预测错误的样本
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        filenames: 文件名列表
        top_k: 返回最常见的 k 个错误类型
    
    Returns:
        错误分析 DataFrame
    """
    errors_df = pd.DataFrame({
        'true': y_true,
        'pred': y_pred,
        'true_name': [CLASS_NAMES[i] for i in y_true],
        'pred_name': [CLASS_NAMES[i] for i in y_pred],
    })
    
    if filenames:
        errors_df['filename'] = filenames
    
    # 只保留错误样本
    errors_df = errors_df[errors_df['true'] != errors_df['pred']]
    
    # 统计错误类型
    error_pairs = errors_df.groupby(['true_name', 'pred_name']).size()
    error_pairs = error_pairs.sort_values(ascending=False).head(top_k)
    
    print("\n🔍 常见错误类型:")
    for (true_name, pred_name), count in error_pairs.items():
        print(f"    {true_name} → {pred_name}: {count} 次")
    
    return errors_df


def compare_models(
    model_results: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    比较不同模型的性能
    
    Args:
        model_results: {model_name: {metric: value}}
    
    Returns:
        比较结果 DataFrame
    """
    df = pd.DataFrame(model_results).T
    df = df.sort_values('accuracy', ascending=False)
    
    print("\n📊 模型比较:")
    print(df.to_string())
    
    return df


def get_ensemble_weights_from_validation(
    results: Dict[str, float],
    power: float = 2.0
) -> Dict[str, float]:
    """
    基于验证结果计算集成权重
    
    Args:
        results: {model_key: accuracy}
        power: 幂次 (更高 = 更偏向高准确率模型)
    
    Returns:
        权重字典
    """
    # 使用准确率的幂次作为权重
    weights = {k: v ** power for k, v in results.items()}
    
    # 归一化
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    
    print("\n🔧 推荐集成权重:")
    for k, v in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v:.4f}")
    
    return weights
