"""
Urban Sound Classification - Optimized Solution
Structure:
1. Config & Setup
2. Dataset & Preprocessing
3. Model Architecture
4. Training Loop (with Scheduler)
5. Inference & Submission
"""

import os
import random
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

# ==========================================
# 1. 配置与全局设置 (Configuration)
# ==========================================
BASE_DIR = Path("C://Users//zhuya//Desktop//Kaggle_Data")  # <--- 【注意】请修改这里为你的实际路径
TRAIN_CSV = BASE_DIR / "metadata" / "kaggle_train.csv"
TEST_CSV = BASE_DIR / "metadata" / "kaggle_test.csv"
AUDIO_DIR = BASE_DIR / "audio"

CONFIG = {
    "SR": 22050,
    "N_MELS": 128,
    "DURATION": 4.0,       # 统一时长 4s
    "MAX_LEN": 173,        # 128x173 的图像尺寸
    "BATCH_SIZE": 32,
    "EPOCHS": 25,          # 稍微增加 Epoch
    "LR": 0.001,
    "SEED": 42,            # 随机种子
    "DEVICE": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
}

def seed_everything(seed):
    """固定随机种子，保证结果可复现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 2. 数据处理与加载 (Dataset)
# ==========================================
# ==========================================
# 新增: SpecAugment 函数
# ==========================================
def spec_augment(spec_img, num_mask=2, freq_masking=15, time_masking=30):
    """
    spec_img: Tensor (1, n_mels, time_steps)
    """
    augmented_spec = spec_img.clone()
    _, n_mels, time_steps = augmented_spec.shape
    
    # 1. Frequency Masking (横条遮挡)
    for _ in range(num_mask):
        f = random.randint(0, freq_masking)
        f0 = random.randint(0, n_mels - f)
        augmented_spec[:, f0:f0+f, :] = 0.0

    # 2. Time Masking (竖条遮挡)
    for _ in range(num_mask):
        t = random.randint(0, time_masking)
        t0 = random.randint(0, time_steps - t)
        augmented_spec[:, :, t0:t0+t] = 0.0
        
    return augmented_spec

def mixup_data(x, y, alpha=0.2, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class UrbanSoundDataset(Dataset):
    def __init__(self, df, audio_dir, mode='train'):
        self.df = df
        self.audio_dir = audio_dir
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['slice_file_name']

        # 1. 路径逻辑 (保持不变)
        if self.mode == 'train':
            file_path = self.audio_dir / f"fold{row['fold']}" / filename
            label = row['classID']
        else:
            file_path = self.audio_dir / "test" / filename
            label = -1 

        try:
            # 2. 加载音频 (保持不变)
            y, _ = librosa.load(str(file_path), sr=CONFIG["SR"], duration=CONFIG["DURATION"])
            target_len = int(CONFIG["SR"] * CONFIG["DURATION"])
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)), mode='constant')
            else:
                y = y[:target_len]

            # 3. 生成频谱 (保持不变)
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=CONFIG["SR"], n_mels=CONFIG["N_MELS"], 
                fmax=8000, hop_length=512
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # 4. 尺寸对齐 (保持不变)
            current_width = mel_spec_db.shape[1]
            if current_width < CONFIG["MAX_LEN"]:
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, CONFIG["MAX_LEN"] - current_width)), mode='constant')
            else:
                mel_spec_db = mel_spec_db[:, :CONFIG["MAX_LEN"]]

            # 5. 归一化 (保持不变)
            delta = mel_spec_db.max() - mel_spec_db.min() + 1e-8
            image = (mel_spec_db - mel_spec_db.min()) / delta
            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

            # --- [新增优化] SpecAugment ---
            # 只有在训练模式下才做增强！验证集和测试集不要做！
            if self.mode == 'train':
                # 以 50% 的概率应用增强
                if random.random() < 0.5:
                    image_tensor = spec_augment(image_tensor)
                return image_tensor, torch.tensor(label, dtype=torch.long)
            else:
                return image_tensor, row['id']

        except Exception as e:
            return torch.zeros((1, CONFIG["N_MELS"], CONFIG["MAX_LEN"])), 0

# ==========================================
# 3. 模型定义 (Model)
# ==========================================
class AudioCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AudioCNN, self).__init__()
        
        # 4层卷积块：提取特征
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # 关键：无论前面尺寸如何，这里都压扁成 1x1
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # 防止过拟合
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==========================================
# 4. K-Fold 训练核心逻辑 (Updated)
# ==========================================
def run_k_fold_training():
    seed_everything(CONFIG["SEED"])
    
    # 读取全部元数据
    full_df = pd.read_csv(TRAIN_CSV)
    
    # 官方数据一共分了 8 个 Fold (1, 2, ..., 8)
    total_folds = 8 
    
    # 用于存储每个 Fold 的最佳验证分数
    fold_scores = []

    print(f"Starting {total_folds}-Fold Cross Validation...")

    for fold_id in range(1, total_folds + 1):
        print(f"\n" + "="*20 + f" Training Fold {fold_id} / {total_folds} " + "="*20)
        
        # --- [关键修改] 动态划分数据集 ---
        # 验证集：当前的 fold_id
        # 训练集：除了当前 fold_id 以外的所有数据
        # 注意：严格遵守官方规则，不要打乱重排 (No Reshuffling) 
        val_df = full_df[full_df['fold'] == fold_id].reset_index(drop=True)
        train_df = full_df[full_df['fold'] != fold_id].reset_index(drop=True)

        print(f"Train set: {len(train_df)} | Val set: {len(val_df)}")
        
        # 创建 DataLoader
        # Windows 用户建议 num_workers=0, Linux/Mac 可设为 2 或 4
        num_workers = 0 if os.name == 'nt' else 2
        
        train_loader = DataLoader(UrbanSoundDataset(train_df, AUDIO_DIR, 'train'), 
                                  batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(UrbanSoundDataset(val_df, AUDIO_DIR, 'train'), 
                                batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=num_workers)

        # 初始化一个全新的模型 (每个 Fold 都要重新初始化，不能继承之前的权重) 
        model = AudioCNN(num_classes=10).to(CONFIG["DEVICE"])
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=CONFIG["LR"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        best_fold_score = 0.0
        
        # 训练 Epochs
        for epoch in range(CONFIG["EPOCHS"]):
            model.train()
            train_loss = 0
            
            for images, labels in tqdm(train_loader, desc=f"Fold {fold_id} Ep {epoch+1}", leave=False):
                images, labels = images.to(CONFIG["DEVICE"]), labels.to(CONFIG["DEVICE"])
                
                optimizer.zero_grad()
                # --- [新增优化] Mixup ---
                # 50% 的概率使用 Mixup，或者全程使用
                use_mixup = True # 建议全程开启或设置概率
                
                if use_mixup:
                    mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.2, device=CONFIG["DEVICE"])
                    outputs = model(mixed_images)
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # 验证 (Validation)
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(CONFIG["DEVICE"]), labels.to(CONFIG["DEVICE"])
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            # 计算分数
            acc = accuracy_score(val_labels, val_preds)
            f1 = f1_score(val_labels, val_preds, average='macro')
            weighted_score = 0.8 * acc + 0.2 * f1 # [cite: 38, 39]

            scheduler.step(weighted_score)

            # 保存当前 Fold 的最佳模型
            # 注意文件名加上 fold_id
            if weighted_score > best_fold_score:
                best_fold_score = weighted_score
                torch.save(model.state_dict(), f"best_model_fold_{fold_id}.pth")
        
        print(f"Fold {fold_id} Best Score: {best_fold_score:.4f}")
        fold_scores.append(best_fold_score)

    print("\n" + "="*40)
    print("Training Complete!")
    print(f"Average Score across 8 folds: {np.mean(fold_scores):.4f}")
    print("Individual Scores:", [f"{s:.4f}" for s in fold_scores])

# ==========================================
# 5. 集成推理与提交 (Ensemble Inference)
# ==========================================
def generate_ensemble_submission():
    print("\nStarting Ensemble Inference (Soft Voting)...")
    
    test_df = pd.read_csv(TEST_CSV)
    test_loader = DataLoader(
        UrbanSoundDataset(test_df, AUDIO_DIR, mode='test'),
        batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=0
    )

    # 准备一个容器来存储所有模型的预测概率
    # 形状: (测试集数量, 10个类别)
    avg_probs = torch.zeros((len(test_df), 10)).to(CONFIG["DEVICE"])
    
    # 遍历所有 8 个训练好的模型
    total_folds = 8
    models_loaded = 0
    
    for fold_id in range(1, total_folds + 1):
        model_path = f"best_model_fold_{fold_id}.pth"
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found, skipping...")
            continue
            
        print(f"Loading {model_path}...")
        model = AudioCNN(num_classes=10).to(CONFIG["DEVICE"])
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        fold_probs = []
        
        with torch.no_grad():
            for images, ids in tqdm(test_loader, desc=f"Predicting Fold {fold_id}"):
                images = images.to(CONFIG["DEVICE"])
                outputs = model(images)
                # 关键：使用 Softmax 获取概率，而不是直接取 max
                probs = torch.softmax(outputs, dim=1) 
                fold_probs.append(probs)
        
        # 将当前 Fold 的预测概率加到总和中
        avg_probs += torch.cat(fold_probs, dim=0)
        models_loaded += 1

    # 取平均 (其实不除也行，因为 max 不受缩放影响，但为了严谨)
    avg_probs /= models_loaded
    
    # 最终决策：选概率最大的类别
    final_predictions = torch.argmax(avg_probs, dim=1).cpu().numpy()
    
    # 生成提交文件
    # 注意：需要重新映射 ID，因为 DataLoader 可能乱序（虽然 shuffle=False，但为了保险）
    # 在这个 Dataset 实现中，我们直接按顺序返回了 ID，所以顺序是对齐的。
    # 但最稳妥的方式是把 ID 也存下来。
    # 这里我们简化处理，假设顺序一致。
    
    # 更好的方式是重新从 test_loader 拿一次 ID（避免变量作用域问题）
    all_ids = []
    for _, ids in test_loader:
        all_ids.extend(ids)
        
    results = []
    for id_val, pred_val in zip(all_ids, final_predictions):
        results.append({'id': id_val, 'label': pred_val})

    sub_df = pd.DataFrame(results)
    sub_df.to_csv("submission_ensemble_8folds.csv", index=False)
    print(f"✓ Ensemble Submission saved! Used {models_loaded} models.")

if __name__ == "__main__":
    # 1. 跑满 8 个 Fold 的训练
    run_k_fold_training()
    
    # 2. 集合 8 个模型的力量生成结果
    generate_ensemble_submission()