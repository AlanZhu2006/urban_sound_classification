"""
Urban Sound Classification - Configuration
机器学习竞赛项目配置文件

工作流程:
1. preprecess.py: 音频 → Mel频谱图 (.npy)
2. train.py: 异构集成训练 + 伪标签生成 → train_pseudo_v3.csv
3. main.py: 最终推理 + 提交生成
"""
import torch
from pathlib import Path

# ==========================================
# 路径配置
# ==========================================
# 本地路径配置
LOCAL_BASE_DIR = Path(__file__).parent.parent / "Kaggle_Data"
LOCAL_NPY_DIR = LOCAL_BASE_DIR / "processed_npy"
LOCAL_SAVE_DIR = Path(__file__).parent.parent / "models"

# Kaggle 路径配置
KAGGLE_BASE_DIR = Path('/kaggle/input/kaggle-data/Kaggle_Data')
KAGGLE_NPY_DIR = Path('/kaggle/input/processed')
KAGGLE_SAVE_DIR = Path('/kaggle/working/models')

# Kaggle 额外目录
KAGGLE_EXTERNAL_WEIGHTS_DIR = Path('/kaggle/input/final-pseudo')
KAGGLE_PTH_FINAL_DIR = Path('/kaggle/input/pth-final')
KAGGLE_PSEUDO_CSV_DIR = Path('/kaggle/input/new-csv')

# 自动检测运行环境
IS_KAGGLE = Path('/kaggle').exists()

if IS_KAGGLE:
    BASE_DIR = KAGGLE_BASE_DIR
    NPY_DIR = KAGGLE_NPY_DIR
    SAVE_DIR = KAGGLE_SAVE_DIR
    EXTERNAL_WEIGHTS_DIR = KAGGLE_EXTERNAL_WEIGHTS_DIR
    # 伪标签训练使用的 CSV
    TRAIN_CSV_PATH = KAGGLE_PSEUDO_CSV_DIR / "train_pseudo_r3.csv"
else:
    BASE_DIR = LOCAL_BASE_DIR
    NPY_DIR = LOCAL_NPY_DIR
    SAVE_DIR = LOCAL_SAVE_DIR
    EXTERNAL_WEIGHTS_DIR = LOCAL_SAVE_DIR
    # 本地使用原始训练 CSV
    TRAIN_CSV_PATH = BASE_DIR / "metadata" / "kaggle_train.csv"

# 测试集 CSV 路径 (通用)
TEST_CSV_PATH = BASE_DIR / "metadata" / "kaggle_test.csv"

# 确保保存目录存在
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 训练阶段配置 (train.py 使用)
# ==========================================
TRAIN_CONFIG = {
    # 模型架构选择 - 异构集成
    "MODELS": ["CRNN", "EffNet"],
    
    # 多分辨率设置
    "FINAL_RESOLUTIONS": [128, 96, 64],
    
    # 训练超参数
    "BATCH_SIZE": 64,
    "EPOCHS": 50,
    "PATIENCE": 12,
    "LR": 1e-3,
    "MIN_LR": 1e-6,
    "SEED": 2024,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "NUM_WORKERS": 2,
    
    # 数据增强
    "USE_CUTMIX": True,
    "CUTMIX_PROB": 0.4,
    "MIXUP_PROB": 0.3,
    "NOISE_LEVEL": 0.015,
    
    # TTA (Test-Time Augmentation)
    "TTA_SHIFTS": [0, -4, 4],
    
    # 伪标签
    "ENABLE_PSEUDO": True,
    "PSEUDO_THRESHOLD": 0.90,
    
    # Hill Climbing 优化
    "HILL_CLIMBING_CYCLES": 1000,
}

# ==========================================
# 推理阶段配置 (main.py 使用)
# ==========================================
INFERENCE_CONFIG = {
    # 模型架构 - 仅使用 CRNN
    "MODELS": ["CRNN"],
    "BASE_RES": 128,
    
    # 密集多分辨率集成
    "FINAL_RESOLUTIONS": [128, 160, 192, 256, 320],
    
    # 训练超参数
    "BATCH_SIZE": 64,
    "EPOCHS": 55,
    "PATIENCE": 12,
    "LR": 1e-3,
    "MIN_LR": 1e-6,
    "SEED": 2024,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "NUM_WORKERS": 2,
    
    # 数据增强
    "USE_CUTMIX": True,
    "CUTMIX_PROB": 0.4,
    "MIXUP_PROB": 0.3,
    "NOISE_LEVEL": 0.015,
    
    # TTA
    "TTA_SHIFTS": [0, -4, 4],
}

# 默认配置 (向后兼容)
CONFIG = INFERENCE_CONFIG.copy()

# 默认配置 (向后兼容)
CONFIG = INFERENCE_CONFIG.copy()

# ==========================================
# 音频处理配置
# ==========================================
AUDIO_CONFIG = {
    "SR": 22050,          # 采样率
    "N_MELS": 128,        # Mel 频谱维度
    "DURATION": 4.0,      # 音频长度（秒）
    "MAX_LEN": 173,       # 特征最大长度
    "FMAX": 8000,         # 最大频率
    "HOP_LENGTH": 512,    # 跳跃长度
}

# ==========================================
# 类别信息
# ==========================================
NUM_CLASSES = 10
CLASS_NAMES = [
    "air_conditioner",    # 0
    "car_horn",           # 1
    "children_playing",   # 2
    "dog_bark",           # 3
    "drilling",           # 4
    "engine_idling",      # 5
    "gun_shot",           # 6
    "jackhammer",         # 7
    "siren",              # 8
    "street_music"        # 9
]

CLASS_NAME_MAP = {i: name for i, name in enumerate(CLASS_NAMES)}

# ==========================================
# 后处理权重配置
# ==========================================
# 类别权重 (用于处理类别不平衡和常见混淆)
# 基于验证集分析:
# - AirConditioner(0): 提权 (常被误判为 Engine)
# - Children(2): 降权 (Recall高但Precision一般)
# - Jackhammer(7): 降权 (常误判其他类)
# - StreetMusic(9): 降权 (过度预测)
CLASS_WEIGHTS = [1.85, 1.00, 0.85, 0.95, 1.25, 1.10, 1.00, 0.65, 1.35, 0.55]

# 分辨率权重 (密集集成时使用)
RES_WEIGHTS = {
    128: 0.6,   # 降权：防止过拟合，但准确度不如大图
    160: 0.9,   
    192: 1.3,   
    256: 1.5,   # 最高权重
    320: 1.2    
}

# train.py 阶段使用的分辨率权重
TRAIN_RES_WEIGHTS = {
    64: 0.95,
    96: 1.08,
    128: 1.15,
}

# ==========================================
# 混淆修正配置
# ==========================================
CONFUSION_CORRECTIONS = {
    # Engine (5) vs Air_conditioner (0)
    "engine_air": {
        "pred_class": 5,
        "target_class": 0,
        "threshold": 0.60,
        "boost": 1.5
    },
    # Street_music (9) vs Children (2)
    "music_children": {
        "pred_class": 9,
        "target_class": 2,
        "threshold": 0.70,
        "boost": 1.5
    },
    # Street_music (9) vs Dog (3)
    "music_dog": {
        "pred_class": 9,
        "target_class": 3,
        "threshold": 0.70,
        "boost": 1.5
    },
    # Jackhammer (7) vs Drilling (4)
    "jackhammer_drilling": {
        "pred_class": 7,
        "target_class": 4,
        "threshold": 0.75,
        "boost": 1.4
    }
}
