"""
Urban Sound Classification - Ultra Fast Version
Optimizations: 
1. RAM Cache (No Disk IO)
2. GPU Augmentation
3. Large Batch Size + AMP (Mixed Precision) 
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
# 引入混合精度
from torch.cuda.amp import autocast, GradScaler

# ==========================================
# 1. 配置 (Configuration)
# ==========================================
BASE_DIR = Path("C://Users//zhuya//Desktop//Kaggle_Data")
TRAIN_CSV = BASE_DIR / "metadata" / "kaggle_train.csv"
TEST_CSV = BASE_DIR / "metadata" / "kaggle_test.csv"
NPY_DIR = BASE_DIR / "processed_npy" 

CONFIG = {
    "N_MELS": 128,
    "MAX_LEN": 173,
    
    # --- [提速修改 1] 加大 Batch Size ---
    # 32 -> 128 (如果显存爆了，改回 64)
    "BATCH_SIZE": 64, 
    
    "EPOCHS": 10,
    "LR": 0.001,
    "SEED": 42,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 2. Augmentation Helpers (保持不变)
# ==========================================
def spec_augment(spec_img, num_mask=2, freq_masking=15, time_masking=30):
    augmented = spec_img.clone()
    batch_size, _, n_mels, time_steps = augmented.shape
    for _ in range(num_mask):
        f = random.randint(0, freq_masking)
        f0 = random.randint(0, n_mels - f)
        augmented[:, :, f0:f0+f, :] = 0.0
    for _ in range(num_mask):
        t = random.randint(0, time_masking)
        t0 = random.randint(0, time_steps - t)
        augmented[:, :, :, t0:t0+t] = 0.0
    return augmented

def mixup_data(x, y, alpha=0.2, device='cuda'):
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

# ==========================================
# 3. Dataset (RAM Cache) (保持不变)
# ==========================================
class UrbanSoundDataset(Dataset):
    def __init__(self, df, data_dir, mode='train'):
        self.df = df
        self.mode = mode
        self.cached_data = []
        self.cached_labels = []
        print(f"Loading {len(df)} samples into RAM ({mode})...")
        for idx, row in tqdm(df.iterrows(), total=len(df), leave=False):
            filename = row['slice_file_name'] + ".npy"
            if mode == 'train':
                file_path = data_dir / f"fold{row['fold']}" / filename
                label = row['classID']
                self.cached_labels.append(label)
            else:
                file_path = data_dir / "test" / filename
            try:
                mel_spec_db = np.load(file_path)
                delta = mel_spec_db.max() - mel_spec_db.min() + 1e-8
                image = (mel_spec_db - mel_spec_db.min()) / delta
                self.cached_data.append(torch.tensor(image, dtype=torch.float32).unsqueeze(0))
            except Exception:
                self.cached_data.append(torch.zeros((1, 128, 173)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_tensor = self.cached_data[idx]
        if self.mode == 'train':
            label = self.cached_labels[idx]
            return image_tensor, torch.tensor(label, dtype=torch.long)
        else:
            return image_tensor, idx # 返回索引作为ID

# ==========================================
# 4. Model (保持不变)
# ==========================================
class AudioCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AudioCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ==========================================
# 5. K-Fold Training Loop (AMP + Early Stopping)
# ==========================================
def run_k_fold_training():
    seed_everything(CONFIG["SEED"])
    full_df = pd.read_csv(TRAIN_CSV)
    
    # 你可以改回 8 折
    total_folds = 1
    fold_scores = []
    
    # --- [提速修改 2] 初始化混合精度 Scaler ---
    scaler = GradScaler()
    
    print(f"🚀 Starting {total_folds}-Fold Training (AMP + Batch {CONFIG['BATCH_SIZE']})...")

    for fold_id in range(1, total_folds + 1):
        print(f"\n=== Fold {fold_id}/{total_folds} ===")
        val_df = full_df[full_df['fold'] == fold_id].reset_index(drop=True)
        train_df = full_df[full_df['fold'] != fold_id].reset_index(drop=True)

        train_loader = DataLoader(UrbanSoundDataset(train_df, NPY_DIR, 'train'), 
                                  batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=0)
        val_loader = DataLoader(UrbanSoundDataset(val_df, NPY_DIR, 'train'), 
                                batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=0)

        model = AudioCNN(num_classes=10).to(CONFIG["DEVICE"])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=CONFIG["LR"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        best_score = 0.0
        early_stop_counter = 0 # 早停计数器
        early_stop_patience = 7 # 7轮不涨就停
        
        for epoch in range(CONFIG["EPOCHS"]):
            model.train()
            
            for images, labels in tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False):
                images, labels = images.to(CONFIG["DEVICE"]), labels.to(CONFIG["DEVICE"])
                
                # GPU Augmentation
                if random.random() < 0.5:
                    images = spec_augment(images)

                optimizer.zero_grad()
                
                # --- [提速修改 3] 混合精度训练 ---
                with autocast():
                    # Mixup
                    if True: 
                        mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.2, device=CONFIG["DEVICE"])
                        outputs = model(mixed_images)
                        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                
                # Scaler Backward
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # Validation
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(CONFIG["DEVICE"]), labels.to(CONFIG["DEVICE"])
                    # 推理时不需要 autocast，但为了省显存也可以加
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            acc = accuracy_score(val_labels, val_preds)
            f1 = f1_score(val_labels, val_preds, average='macro')
            weighted_score = 0.8 * acc + 0.2 * f1
            
            scheduler.step(weighted_score)

            # --- [提速修改 4] 保存与早停逻辑 ---
            if weighted_score > best_score:
                best_score = weighted_score
                torch.save(model.state_dict(), f"best_model_fold_{fold_id}.pth")
                early_stop_counter = 0 # 重置计数器
            else:
                early_stop_counter += 1
                
            if early_stop_counter >= early_stop_patience:
                print(f"🛑 Early stopping at epoch {epoch+1} (Score: {weighted_score:.4f})")
                break
        
        print(f"✅ Fold {fold_id} Best Score: {best_score:.4f}")
        fold_scores.append(best_score)

    print(f"\nAverage Score: {np.mean(fold_scores):.4f}")

# ==========================================
# 6. Submission (保持不变)
# ==========================================
def generate_ensemble_submission():
    print("\n🚀 Generating Submission...")
    test_df = pd.read_csv(TEST_CSV)
    
    # Inference batch size 也可以加大
    test_loader = DataLoader(UrbanSoundDataset(test_df, NPY_DIR, 'test'), 
                             batch_size=CONFIG["BATCH_SIZE"]*2, shuffle=False, num_workers=0)

    avg_probs = torch.zeros((len(test_df), 10)).to(CONFIG["DEVICE"])
    models_loaded = 0
    
    # 遍历所有保存的模型
    for fold_id in range(1, 9):
        path = f"best_model_fold_{fold_id}.pth"
        if not os.path.exists(path): continue
        
        model = AudioCNN(num_classes=10).to(CONFIG["DEVICE"])
        model.load_state_dict(torch.load(path))
        model.eval()
        
        fold_probs = []
        with torch.no_grad():
            for images, _ in tqdm(test_loader, desc=f"Fold {fold_id}"):
                images = images.to(CONFIG["DEVICE"])
                fold_probs.append(torch.softmax(model(images), dim=1))
        
        avg_probs += torch.cat(fold_probs, dim=0)
        models_loaded += 1

    if models_loaded == 0:
        print("❌ 错误：没有找到模型文件！")
        return

    final_preds = torch.argmax(avg_probs / models_loaded, dim=1).cpu().numpy()
    
    sub_df = pd.DataFrame({
        'ID': range(len(test_df)), 
        'TARGET': final_preds
    })
    
    sub_df.to_csv("submission_fixed.csv", index=False)
    print(f"✓ 成功生成 submission_fixed.csv (包含 ID, TARGET), 共 {len(sub_df)} 行。")

if __name__ == "__main__":
    if not NPY_DIR.exists():
        print("❌ 请先运行预处理脚本生成 .npy 文件！")
    else:
        run_k_fold_training()
        generate_ensemble_submission()