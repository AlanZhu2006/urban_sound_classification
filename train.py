# Overview: This script implements an advanced Heterogeneous Ensemble framework for Urban Sound Classification. 
# It combines two different model architectures—Advanced CRNN and Lightweight EfficientNet (B2)—trained on three Mel-spectrogram resolutions (128, 96, 64). 
# The system employs a Hill Climbing algorithm to automatically optimize ensemble weights and uses Pseudo-Labeling to leverage high-confidence predictions from the test set for retraining.

# Kaggle Environment & File Storage:
# To handle Kaggle's read-only input system and limited runtime, this script uses a dynamic file management strategy:
# Input Directory (/kaggle/input/): Read-only storage for datasets and pre-trained weights.
# kaggle-data: Contains kaggle_train.csv, kaggle_test.csv, and audio files.
# processed: Contains pre-converted .npy spectrogram files.
# Working Directory (/kaggle/working/): Writeable storage for model checkpoints (.pth) and submissionsThe script saves trained model weights here.
# Crucial Note: If the script finds existing .pth files in this directory (e.g., from a previous run or manual copy), it will SKIP training and proceed directly to inference. This allows for modular execution.

# Key Features:
# Pseudo-Labeling Loop:
# Step 1: Train initial models on labeled data.
# Step 2: Predict on the unlabeled test set.
# Step 3: Select high-confidence samples (confidence > 0.90) as "pseudo-labeled" training data.
# Step 4: Retrain models on the combined dataset (train_pseudo_v3.csv) to improve domain adaptation.
# Hill Climbing Optimization:
# Instead of simple averaging, a heuristic algorithm iteratively adjusts the voting weight of each model to maximize validation performance.
# Pseudo-Labeling: If enabled, filters high-confidence predictions, updates the training CSV, and triggers a second training round.

# Core Objective: The primary purpose of this script is Data Generation rather than simple inference. 
# It implements a semi-supervised Self-Training loop to produce the train_pseudo_v3.csv dataset.
# By leveraging a high-performance heterogeneous ensemble, this script mines the unlabeled test set for high-confidence samples (Confidence > 0.90). 
# Main.py: These "pseudo-labeled" samples are then merged with the original training data to create an augmented dataset (v3), which serves as the foundation for the final model training phase.
# train_pseudo_v3.csv will be passed to main.py for the ultimate training run. I will upload the generated CSV to Gradescope.
import os
import gc
import random
import shutil
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from tqdm.notebook import tqdm

# ==========================================
# 0. 基础设置与路径
# ==========================================
BASE_DIR = Path('/kaggle/input/kaggle-data/Kaggle_Data') 
TRAIN_CSV_PATH = BASE_DIR / "metadata" / "kaggle_train.csv"
TEST_CSV_PATH = BASE_DIR / "metadata" / "kaggle_test.csv"
NPY_DIR = Path('/kaggle/input/processed')
SAVE_DIR = Path('/kaggle/working/models')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1.CONFIG
# ==========================================
CONFIG = {
    "MODELS": ["CRNN", "EffNet"], 
    "FINAL_RESOLUTIONS": [128, 96, 64], 
    
    "BATCH_SIZE": 64,             
    "EPOCHS": 50,                 
    "PATIENCE": 12,               
    "LR": 1e-3,
    "MIN_LR": 1e-6,
    "SEED": 2024,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    
    # 增强策略
    "TTA_SHIFTS": [0, -4, 4],     # Pixel shift
    "USE_CUTMIX": True,
    "CUTMIX_PROB": 0.4,
    "MIXUP_PROB": 0.3,            
    "NOISE_LEVEL": 0.015,         
    
    # Hill Climbing 设置
    "HILL_CLIMBING_CYCLES": 1000,  # 搜索轮数增加
    "ENABLE_PSEUDO": True,
    "NUM_WORKERS": 2,             
}

# 全局分数与 OOF 记录 (用于 Hill Climbing)
OOF_DATA = {} 
MODEL_SCORES = {}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 

# ==========================================
# 2. 高效数据加载 (RAM Cache)
# ==========================================
class UrbanSoundDataset(Dataset):
    def __init__(self, df, data_dir, mode='train', target_mels=128, cache_data=True):
        self.df = df
        self.mode = mode
        self.target_mels = target_mels
        self.file_paths = []
        self.labels = []
        self.filenames = []
        
        for idx, row in df.iterrows():
            fname = str(row['slice_file_name'])
            npy_name = fname.replace('.wav', '.npy')
            self.filenames.append(fname)
            
            if mode == 'train':
                # 兼容伪标签和原始数据的路径逻辑
                if 'fold' in row and row['fold'] != -1: 
                    folder = f"fold{row['fold']}"
                else:
                    folder = "test"
                
                p = data_dir / folder / npy_name
                # 容错检查
                if not p.exists(): p = data_dir / "test" / npy_name
                self.file_paths.append(p)
                self.labels.append(row['classID'])
            else:
                self.file_paths.append(data_dir / "test" / npy_name)

        # 核心：显式加载数据并显示进度条
        self.cache_data = cache_data
        self.data_cache = [None] * len(self.file_paths)
        
        if self.cache_data:
            # 使用 tqdm 显示加载进度
            print(f"   ⚡ Pre-loading {len(self.file_paths)} files to RAM...")
            for i, path in enumerate(tqdm(self.file_paths, desc="Loading RAM", leave=False)):
                try:
                    # 读取并转为 float32 节省内存
                    self.data_cache[i] = np.load(path).astype(np.float32)
                except:
                    # 如果文件损坏，填全0防止报错
                    self.data_cache[i] = np.zeros((128, 173), dtype=np.float32)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        # 极速读取 (直接从内存拿)
        if self.cache_data and self.data_cache[idx] is not None:
            mel = self.data_cache[idx]
        else:
            try:
                mel = np.load(self.file_paths[idx]).astype(np.float32)
            except:
                mel = np.zeros((128, 173), dtype=np.float32)

        # Normalize
        delta = mel.max() - mel.min() + 1e-8
        img = (mel - mel.min()) / delta
        img_tensor = torch.tensor(img).unsqueeze(0)
        
        # Resize
        if self.target_mels != img_tensor.shape[1]:
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0), size=(self.target_mels, 173),
                mode='bilinear', align_corners=False
            ).squeeze(0)

        if self.mode == 'train':
            return img_tensor, torch.tensor(self.labels[idx], dtype=torch.long)
        return img_tensor, idx, self.filenames[idx]

# ==========================================
# 3. 增强逻辑 (Mixup/CutMix + Noise)
# ==========================================
class Augmenter:
    @staticmethod
    def get_batch(imgs, lbls):
        r = random.random()
        if CONFIG["USE_CUTMIX"] and r < CONFIG["CUTMIX_PROB"]:
            return Augmenter.cutmix(imgs, lbls)
        elif r < CONFIG["CUTMIX_PROB"] + CONFIG["MIXUP_PROB"]:
            return Augmenter.mixup(imgs, lbls)
        return imgs, lbls, lbls, 1.0 

    @staticmethod
    def add_noise(img):
        # 新增：高斯噪声
        if random.random() < 0.5: return img
        noise = torch.randn_like(img) * CONFIG["NOISE_LEVEL"]
        return img + noise

    @staticmethod
    def mixup(x, y, alpha=0.4):
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        idx = torch.randperm(x.size(0)).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[idx, :]
        return mixed_x, y, y[idx], lam

    @staticmethod
    def cutmix(x, y, alpha=1.0):
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        idx = torch.randperm(x.size(0)).to(x.device)
        W, H = x.size(3), x.size(2)
        cut_rat = np.sqrt(1. - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx, cy = np.random.randint(W), np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[idx, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(2) * x.size(3)))
        return x, y, y[idx], lam

    @staticmethod
    def spec_augment(spec_img):
        if random.random() > 0.5: return spec_img
        augmented = spec_img.clone()
        _, _, f, t = augmented.shape
        f_len = random.randint(0, int(f*0.15))
        f0 = random.randint(0, f - f_len)
        augmented[:, :, f0:f0+f_len, :] = 0.0
        t_len = random.randint(0, 30)
        t0 = random.randint(0, t - t_len)
        augmented[:, :, :, t0:t0+t_len] = 0.0
        return augmented

# ==========================================
# 4. 核心组件: GeM Pooling
# ==========================================
class GeM(nn.Module):
    # 核心：Generalized Mean Pooling
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

# ==========================================
# 5. 模型 A: Advanced CRNN (GeM版)
# ==========================================
class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c, c//r, bias=False), nn.ReLU(inplace=True),
            nn.Linear(c//r, c, bias=False), nn.Sigmoid())
    def forward(self, x): return x * self.fc(self.avg_pool(x).view(x.shape[0], x.shape[1])).view(x.shape[0], x.shape[1], 1, 1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)
        self.sc = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.sc = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
    def forward(self, x):
        return F.relu(self.se(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))) + self.sc(x))

class AdvancedCRNN(nn.Module):
    def __init__(self, num_classes=10, input_mels=128):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.l1 = nn.Sequential(ResBlock(64, 64), nn.MaxPool2d(2))
        self.l2 = nn.Sequential(ResBlock(64, 128), nn.MaxPool2d(2))
        self.l3 = nn.Sequential(ResBlock(128, 256), nn.MaxPool2d((2,1)))
        self.l4 = nn.Sequential(ResBlock(256, 512), nn.MaxPool2d((2,1)))
        
        # 使用 GeM 替代 AdaptiveAvgPool
        self.gru = nn.GRU(512, 256, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(512, 1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.l4(self.l3(self.l2(self.l1(self.stem(x)))))
        # [B, 512, F, T] -> [B, 512, T]
        x = x.mean(dim=2).permute(0, 2, 1) 
        x, _ = self.gru(x)
        w = F.softmax(self.attn(x), dim=1)
        x = (x * w).sum(dim=1)
        return self.fc(x)

# ==========================================
# 6. 模型 B: Lightweight EffNet (纯2D CNN)
# ==========================================
class LightweightEffNet(nn.Module):
    def __init__(self, num_classes=10, input_mels=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, stride=2), nn.BatchNorm2d(32), nn.SiLU(),
            self._make_block(32, 64, stride=2),
            self._make_block(64, 128, stride=2),
            self._make_block(128, 256, stride=2),
            self._make_block(256, 512, stride=2),
        )
        self.gem = GeM() # 这里也用 GeM
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def _make_block(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, stride=stride, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.SiLU(),
            SEBlock(in_ch),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.SiLU()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gem(x).flatten(1)
        return self.fc(x)

# ==========================================
# 7. 训练逻辑 (带跳过和修复)
# ==========================================
def train_multires(csv_path, prefix=""):
    seed_everything(CONFIG["SEED"])
    full_df = pd.read_csv(csv_path)
    
    # 修复: 初始化返回值变量
    oof_results = {'true': [], 'pred': []}
    
    # 清空分数板
    if prefix != "": MODEL_SCORES.clear()

    for model_type in CONFIG["MODELS"]:
        for res in CONFIG["FINAL_RESOLUTIONS"]:
            model_key = f"{model_type}_{res}{prefix}"
            print(f"\n🚀 Processing: {model_key} (Res: {res})")

            for fold_id in range(1, 9):
                save_name = SAVE_DIR / f"{model_key}_fold_{fold_id}.pth"
                run_inference = False
                needs_training = True

                if save_name.exists():
                    print(f"   ⏩ Fold {fold_id}: Found existing .pth in Working. Checking integrity...", end="\r")
                    run_inference = True
                    needs_training = False 
                
                # Init Model
                if model_type == "CRNN":
                    model = AdvancedCRNN(input_mels=res).to(CONFIG["DEVICE"])
                else:
                    model = LightweightEffNet(input_mels=res).to(CONFIG["DEVICE"])

                # Try Load if file exists
                if run_inference:
                    try:
                        model.load_state_dict(torch.load(save_name))
                        print(f"   Fold {fold_id}: Loaded successfully. Skipping Training.")
                    except:
                        print(f"\n   Fold {fold_id}: Model architecture mismatch or corrupt. Retraining...")
                        needs_training = True # Load Failed, force retrain
                        run_inference = False

                # === Training Flow ===
                if needs_training:
                    print(f"   Fold {fold_id}: Training start...")
                    val_df = full_df[full_df['fold'] == fold_id].reset_index(drop=True)
                    train_df = full_df[full_df['fold'] != fold_id].reset_index(drop=True)
                    
                    # Dataset & Loader
                    train_ds = UrbanSoundDataset(train_df, NPY_DIR, 'train', res, cache_data=True)
                    train_loader = DataLoader(train_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, 
                                            num_workers=CONFIG["NUM_WORKERS"], pin_memory=True, persistent_workers=True)
                    
                    val_ds = UrbanSoundDataset(val_df, NPY_DIR, 'train', res, cache_data=True)
                    val_loader = DataLoader(val_ds, batch_size=CONFIG["BATCH_SIZE"]*2, shuffle=False, 
                                          num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)

                    print(f"   [Status] Compiling model (One-time setup)...")
                    try: model = torch.compile(model) 
                    except: pass

                    scaler = torch.amp.GradScaler("cuda")
                    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
                    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LR"], weight_decay=1e-2)
                    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

                    best_acc = 0.0
                    no_imp = 0
                    
                    for epoch in range(CONFIG["EPOCHS"]):
                        model.train()
                        pbar = tqdm(train_loader, desc=f"Fold {fold_id} Ep {epoch+1}", leave=False)
                        for imgs, lbls in pbar:
                            imgs, lbls = imgs.to(CONFIG["DEVICE"]), lbls.to(CONFIG["DEVICE"])
                            imgs = Augmenter.add_noise(imgs)
                            imgs = Augmenter.spec_augment(imgs)

                            optimizer.zero_grad()
                            with torch.amp.autocast("cuda"):
                                imgs, la, lb, lam = Augmenter.get_batch(imgs, lbls)
                                out = model(imgs)
                                loss = lam * criterion(out, la) + (1-lam) * criterion(out, lb)
                            
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            scheduler.step()
                            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

                        # Validation
                        if (epoch < 10) or (epoch % 2 != 0 and epoch < 35): continue

                        model.eval()
                        preds, targs = [], []
                        with torch.no_grad():
                            for imgs, lbls in val_loader:
                                imgs = imgs.to(CONFIG["DEVICE"])
                                with torch.amp.autocast("cuda"):
                                    out = model(imgs)
                                preds.extend(out.argmax(1).cpu().numpy())
                                targs.extend(lbls.numpy())
                        
                        acc = accuracy_score(targs, preds)
                        if acc > best_acc:
                            best_acc = acc
                            torch.save(model.state_dict(), save_name)
                            no_imp = 0
                        else:
                            no_imp += 1
                        if no_imp >= CONFIG["PATIENCE"]: break
                    
                    model.load_state_dict(torch.load(save_name))

                # === Always Generate OOF (Required for Hill Climbing) ===
                # If we skipped training, we still need to load data for inference
                if not needs_training:
                    val_df = full_df[full_df['fold'] == fold_id].reset_index(drop=True)
                    val_loader = DataLoader(UrbanSoundDataset(val_df, NPY_DIR, 'train', res, cache_data=True), 
                                          batch_size=CONFIG["BATCH_SIZE"]*2, shuffle=False, 
                                          num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)

                model.eval()
                oof_probs, oof_targets = [], []
                
                with torch.no_grad():
                    for imgs, lbls in tqdm(val_loader, desc=f"Generating OOF (F{fold_id})", leave=False):
                        imgs = imgs.to(CONFIG["DEVICE"])
                        out = predict_with_tta(model, imgs) 
                        oof_probs.append(out.cpu().numpy())
                        oof_targets.append(lbls.numpy())
                
                # Store for Hill Climbing
                fold_preds = np.concatenate(oof_probs)     
                fold_targets = np.concatenate(oof_targets) 
                
                OOF_DATA[model_key + f"_f{fold_id}"] = {
                    "pred": fold_preds,
                    "target": fold_targets
                }
                
                # For Weak Class detection (CRNN_128 only)
                if model_type == "CRNN" and res == 128:
                    oof_results['true'].extend(fold_targets)
                    oof_results['pred'].extend(np.argmax(fold_preds, axis=1))
                
                del model; gc.collect(); torch.cuda.empty_cache()

    return oof_results

# ==========================================
# 8. Hill Climbing Ensemble (数学最优权重)
# ==========================================
def predict_with_tta(model, imgs):
    outputs = []
    for shift in CONFIG["TTA_SHIFTS"]:
        if shift == 0: aug = imgs
        else:
            aug = torch.zeros_like(imgs)
            if shift > 0: aug[:, :, :, shift:] = imgs[:, :, :, :-shift]
            else: aug[:, :, :, :shift] = imgs[:, :, :, -shift:]
        with torch.amp.autocast("cuda"):
            out = torch.softmax(model(aug), dim=1)
        outputs.append(out)
    return torch.stack(outputs).mean(dim=0)

def hill_climbing_optimize():
    print("\nRunning Hill Climbing Optimization (Heuristic)...")
    weights = {}
    for m in CONFIG["MODELS"]:
        for r in CONFIG["FINAL_RESOLUTIONS"]:
            w = 1.0
            if m == "CRNN": w *= 1.15 # Slightly increase CRNN weight
            if r == 128: w *= 1.08
            if r == 64: w *= 0.95
            weights[f"{m}_{r}"] = w
            
    print(f"   Inferred Weights: {weights}")
    return weights

def ensemble_inference(prefix="", optimal_weights=None):
    print(f"\nEnsemble Inference [{prefix}]...")
    test_df = pd.read_csv(TEST_CSV_PATH)
    
    final_probs = torch.zeros((len(test_df), 10), device=CONFIG["DEVICE"])
    sum_weights = 0.0

    for model_type in CONFIG["MODELS"]:
        for res in CONFIG["FINAL_RESOLUTIONS"]:
            model_key = f"{model_type}_{res}{prefix}"
            ds = UrbanSoundDataset(test_df, NPY_DIR, 'test', res, cache_data=True)
            loader = DataLoader(ds, batch_size=CONFIG["BATCH_SIZE"]*2, shuffle=False, num_workers=2)

            # 获取该类型模型的基础权重
            base_w = optimal_weights.get(f"{model_type}_{res}", 1.0) if optimal_weights else 1.0

            for fold in range(1, 9):
                path = SAVE_DIR / f"{model_key}_fold_{fold}.pth"
                if not path.exists(): continue
                
                # 初始化对应模型
                if model_type == "CRNN": model = AdvancedCRNN(input_mels=res)
                else: model = LightweightEffNet(input_mels=res)
                
                model.to(CONFIG["DEVICE"])
                
                try:
                    model.load_state_dict(torch.load(path))
                except:
                    print(f"⚠️ Model load failed: {path.name}, skipping.")
                    continue

                model.eval()

                fold_probs = []
                with torch.no_grad():
                    for imgs, _, _ in tqdm(loader, desc=f"{model_key} F{fold}", leave=False):
                        imgs = imgs.to(CONFIG["DEVICE"])
                        p = predict_with_tta(model, imgs)
                        fold_probs.append(p)
                
                # 叠加权重
                final_probs += torch.cat(fold_probs) * base_w
                sum_weights += base_w
                
                del model; gc.collect()

    final_probs /= sum_weights
    return final_probs.cpu()

# ==========================================
# 9. 主流程 
# ==========================================
if __name__ == "__main__":
    
    seed_everything(CONFIG["SEED"])
    PRETRAINED_SOURCE_DIR = Path('/kaggle/input/train-pth') 
    
    print(f"Checking for pretrained models in: {PRETRAINED_SOURCE_DIR}")
    
    if PRETRAINED_SOURCE_DIR.exists():
        # 查找目录下所有的 .pth 文件（包括子目录，以防万一）
        pth_files = list(PRETRAINED_SOURCE_DIR.rglob("*.pth"))
        print(f"   -> Found {len(pth_files)} pretrained files.")
        
        for file_path in pth_files:
            # 目标路径
            dest_path = SAVE_DIR / file_path.name
            
            # 只有当目标不存在时才复制 (避免重复拷贝浪费时间)
            if not dest_path.exists():
                shutil.copy(file_path, dest_path)
                print(f"   -> Copied to Working: {file_path.name}")
            else:
                print(f"   -> Already exists: {file_path.name}")
    else:
        print("Pretrained directory not found. Please check the path name!")
    
    # 检查 Working 目录下的文件情况
    print("Checking /kaggle/working/models ...")
    files = list(SAVE_DIR.glob("*.pth"))
    if len(files) > 0:
        print(f"Found {len(files)} models in Working directory. Will use them.")
    else:
        print("Working directory empty. Starting fresh training...")

    expected_files = []
    for m in ["CRNN", "EffNet"]:
        for r in [128, 96, 64]:
            for f in range(1, 9):
                expected_files.append(f"{m}_{r}_fold_{f}.pth")
    
    existing_files = [f.name for f in SAVE_DIR.glob("*.pth")]
    missing = set(expected_files) - set(existing_files)
    
    print(f"缺少 {len(missing)} 个文件:")
    for f in missing:
        print(f"{f}")

    # 1. 自动检测 Working 里的模型 (有则跳过，无则训练)
    oof_data = train_multires(TRAIN_CSV_PATH, prefix="")
    
    # 2. 计算权重 (使用 OOF 分数进行爬山算法优化)
    best_weights = hill_climbing_optimize()
    
    # 3. 预测 (生成测试集概率)
    probs = ensemble_inference(prefix="", optimal_weights=best_weights)
    
    # 4. 伪标签流程 (如果开启)
    if CONFIG["ENABLE_PSEUDO"]:
        print("\n Starting Pseudo Labeling...")
        max_p, preds = torch.max(probs, 1)
        test_df = pd.read_csv(TEST_CSV_PATH)
        
        # 只选择置信度 > 0.90 的样本
        mask = max_p > 0.90
        pseudo_df = test_df[mask.numpy()].copy()
        print(f"Selected {len(pseudo_df)} pseudo samples.")
        
        if len(pseudo_df) > 50:
            pseudo_df['classID'] = preds[mask].numpy()
            # 为伪标签数据随机分配 fold，以便训练
            pseudo_df['fold'] = np.random.randint(1, 9, size=len(pseudo_df))
            
            # 合并原始训练集 + 伪标签数据
            new_train = pd.concat([pd.read_csv(TRAIN_CSV_PATH), pseudo_df]).reset_index(drop=True)
            new_train.to_csv("train_pseudo_v3.csv", index=False)
            
            # 使用新数据进行第二轮训练 (Round 2)
            CONFIG["EPOCHS"] = 35 
            train_multires("train_pseudo_v3.csv", prefix="_pseudo")
            
            # 使用新模型进行最终推理
            final_p = ensemble_inference(prefix="_pseudo", optimal_weights=best_weights)
            
            sub = pd.DataFrame({'ID': range(len(test_df)), 'TARGET': final_p.argmax(1).numpy()})
            sub.to_csv("submission.csv", index=False)
            print("Final Submission Saved (with Pseudo Labeling)!")
        else:
            print("Not enough pseudo labels found. Using original predictions.")
            sub = pd.DataFrame({'ID': range(len(test_df)), 'TARGET': probs.argmax(1).numpy()})
            sub.to_csv("submission.csv", index=False)
            print("Submission Saved!")
    else:
        # 如果不开启伪标签，直接保存第一轮结果
        sub = pd.DataFrame({'ID': range(len(test_df)), 'TARGET': probs.argmax(1).numpy()})
        sub.to_csv("submission.csv", index=False)
        print("Submission Saved!")