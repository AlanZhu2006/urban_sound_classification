"""
Urban Sound Classification 项目包

项目流程:
1. preprecess.py: 音频 → Mel频谱图 (.npy)
2. train.py: 异构集成训练 (CRNN+EffNet) + 伪标签生成 → train_pseudo_v3.csv
3. main.py: 密集多分辨率集成推理 + 后处理 → submission.csv

模块说明:
- config: 配置参数 (训练/推理阶段分离)
- preprocessing: 音频预处理
- dataset: 数据集类 (RAM缓存加速)
- augmentation: 数据增强 (Mixup/CutMix/SpecAugment)
- models: 模型定义 (CRNN/EffNet)
- train: 训练函数 (8折交叉验证)
- inference: 推理函数 (TTA + 后处理)
- evaluate: 评估函数
- utils: 工具函数
"""
from .config import (
    CONFIG, TRAIN_CONFIG, INFERENCE_CONFIG,
    AUDIO_CONFIG, CLASS_NAMES, CLASS_NAME_MAP,
    CLASS_WEIGHTS, RES_WEIGHTS, CONFUSION_CORRECTIONS,
    SAVE_DIR, NPY_DIR, TRAIN_CSV_PATH, TEST_CSV_PATH,
    IS_KAGGLE
)
from .utils import seed_everything, cleanup_memory, get_device
from .preprocessing import preprocess_all
from .dataset import UrbanSoundDataset, create_data_loaders, create_test_loader
from .augmentation import Augmenter
from .models import create_model, AdvancedCRNN, LightweightEffNet, FocalLoss, GeM, SEBlock, ResBlock
from .train import train_model, train_multi_resolution, train_one_fold, create_pseudo_labels
from .inference import (
    ensemble_inference, generate_submission, 
    predict_with_tta, apply_class_weights, 
    apply_power_sharpening, apply_confusion_corrections
)
from .evaluate import local_validation

__version__ = "1.0.0"
__all__ = [
    # Config
    "CONFIG", "TRAIN_CONFIG", "INFERENCE_CONFIG",
    "AUDIO_CONFIG", "CLASS_NAMES", "CLASS_NAME_MAP",
    "CLASS_WEIGHTS", "RES_WEIGHTS", "CONFUSION_CORRECTIONS",
    "SAVE_DIR", "NPY_DIR", "TRAIN_CSV_PATH", "TEST_CSV_PATH", "IS_KAGGLE",
    # Utils
    "seed_everything", "cleanup_memory", "get_device",
    # Preprocessing
    "preprocess_all",
    # Dataset
    "UrbanSoundDataset", "create_data_loaders", "create_test_loader",
    # Augmentation
    "Augmenter",
    # Models
    "create_model", "AdvancedCRNN", "LightweightEffNet", "FocalLoss", "GeM", "SEBlock", "ResBlock",
    # Training
    "train_model", "train_multi_resolution", "train_one_fold", "create_pseudo_labels",
    # Inference
    "ensemble_inference", "generate_submission",
    "predict_with_tta", "apply_class_weights",
    "apply_power_sharpening", "apply_confusion_corrections",
    # Evaluation
    "local_validation",
]
