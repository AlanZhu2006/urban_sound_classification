# Overview: This script implements a Dense Multi-Resolution Ensemble strategy for Urban Sound Classification. 
# Instead of relying on a single model, it aggregates predictions from Advanced CRNN models trained on five different Mel-spectrogram resolutions (128, 160, 192, 256, 320). 
# It includes a robust post-processing pipeline with Test-Time Augmentation (TTA), Power Sharpening, and dynamic conflict resolution to correct specific class confusions.

# Kaggle Environment & File Storage (Crucial)
# To ensure reproducibility and bypass the 12-hour Kaggle runtime limit, this script uses a "Load Pre-trained or Train from Scratch" logic.

# Input Directory (/kaggle/input/): This is read-only.
# pth-final and final-pseudo: Contains your pre-trained model weights (e.g., .pth files).
# new-csv: Contains augmented training CSV files on the basis of Kaggle_train.csv with pseudo-labels from train.py.
# kaggle-data: Contains the competition metadata and audio.
# processed: Contains the pre-converted .npy spectrogram files produced by preprocess.npy.

# Working Directory (/kaggle/working/): This is writeable. The script automatically copies weights from Input to Working at the start.
# The model loads weights from here. If a weight is missing, it triggers the training loop automatically.

# File Naming Conventions:
# The script relies on a strict naming pattern to identify and load weights. The logic inside train_model looks for files matching:

# CRNN_{Resolution}_{Prefix}_fold_{FoldID}.pth eg. CRNN_192_final_pseudo_fold_1.pth 
# These pth file should be included in the pth zip I have uploaded to Gradescope.

# TRAIN_CSV_PATH = Path('/kaggle/input/new-csv/train_pseudo_r3.csv') a file produced by train.py. This is a new csv produced on the basis of Kaggle_train.csv. Augmented training dataset generated via 3 rounds of iterative self-training, incorporating high-confidence test predictions as pseudo-labels.

# Important: this script has to be run in a Kaggle Notebook environment where the dataset is mounted!!!


import os
import gc
import random
import time
import shutil  
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm

# ==========================================
# 0. 基础设置
# ==========================================
EXTERNAL_WEIGHTS_DIR = Path('/kaggle/input/final-pseudo')

BASE_DIR = Path('/kaggle/input/kaggle-data/Kaggle_Data')
TRAIN_CSV_PATH = Path('/kaggle/input/new-csv/train_pseudo_r3.csv')
TEST_CSV_PATH = BASE_DIR / "metadata" / "kaggle_test.csv"
NPY_DIR = Path('/kaggle/input/processed')

SAVE_DIR = Path('/kaggle/working/models')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. 策略配置
# ==========================================
CONFIG = {
    "MODELS": ["CRNN"],
    "BASE_RES": 128,
    # 覆盖所有分辨率，程序会自动去 input 找权重
    "FINAL_RESOLUTIONS": [128, 160, 192, 256, 320],
    "BATCH_SIZE": 64,
    "EPOCHS": 55,
    "PATIENCE": 12,
    "LR": 1e-3,
    "MIN_LR": 1e-6,
    "SEED": 2024,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "TTA_SHIFTS": [0, -4, 4],
    "USE_CUTMIX": True,
    "CUTMIX_PROB": 0.4,
    "MIXUP_PROB": 0.3,
    "NOISE_LEVEL": 0.015,
    "NUM_WORKERS": 2,
}

# ==========================================
# 2. Dataset
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
                if 'fold' in row and row['fold'] != -1:
                    folder = f"fold{row['fold']}"
                else:
                    folder = "test"
                
                p = data_dir / folder / npy_name
                if not p.exists():
                    p = data_dir / "test" / npy_name
                self.file_paths.append(p)
                self.labels.append(row['classID'])
            else:
                self.file_paths.append(data_dir / "test" / npy_name)

        self.cache_data = cache_data
        self.data_cache = [None] * len(self.file_paths)
        
        if self.cache_data:
            print(f"    Pre-loading {len(self.file_paths)} files...")
            for i, path in enumerate(tqdm(self.file_paths, desc="Loading RAM", leave=False)):
                try:
                    self.data_cache[i] = np.load(path).astype(np.float32)
                except:
                    self.data_cache[i] = np.zeros((128, 173), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.cache_data and self.data_cache[idx] is not None:
            mel = self.data_cache[idx]
        else:
            try:
                mel = np.load(self.file_paths[idx]).astype(np.float32)
            except:
                mel = np.zeros((128, 173), dtype=np.float32)

        delta = mel.max() - mel.min() + 1e-8
        img = (mel - mel.min()) / delta
        img_tensor = torch.tensor(img).unsqueeze(0)
        
        if self.target_mels != img_tensor.shape[1]:
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(self.target_mels, 173),
                mode='bilinear', align_corners=False
            ).squeeze(0)

        if self.mode == 'train':
            return img_tensor, torch.tensor(self.labels[idx], dtype=torch.long)
        return img_tensor, idx, self.filenames[idx]

# ==========================================
# 3. Augmenter
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
        if random.random() < 0.5: return img
        return img + torch.randn_like(img) * CONFIG["NOISE_LEVEL"]

    @staticmethod
    def mixup(x, y, alpha=0.4):
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        idx = torch.randperm(x.size(0)).to(x.device)
        return lam * x + (1 - lam) * x[idx, :], y, y[idx], lam

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
        return x, y, y[idx], 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(2) * x.size(3)))

    @staticmethod
    def spec_augment(spec_img):
        if random.random() > 0.5: return spec_img
        augmented = spec_img.clone()
        f_len = random.randint(0, int(augmented.shape[2]*0.15))
        f0 = random.randint(0, augmented.shape[2] - f_len)
        augmented[:, :, f0:f0+f_len, :] = 0.0
        time_dim = augmented.shape[3]
        t_len = random.randint(0, int(time_dim * 0.15))
        t0 = random.randint(0, time_dim - t_len)
        augmented[:, :, :, t0:t0+t_len] = 0.0
        return augmented

# ==========================================
# 4. Model
# ==========================================
class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c, c//r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c//r, c, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(self.avg_pool(x).view(x.shape[0], x.shape[1])).view(x.shape[0], x.shape[1], 1, 1)

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
        self.gru = nn.GRU(512, 256, batch_first=True, bidirectional=True, dropout=0.2, num_layers=2)
        self.attn = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.l4(self.l3(self.l2(self.l1(self.stem(x)))))
        x = x.mean(dim=2).permute(0, 2, 1)
        x, _ = self.gru(x)
        w = F.softmax(self.attn(x), dim=1)
        x = (x * w).sum(dim=1)
        x = self.dropout(x)
        return self.fc(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        return focal_loss.sum()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# ==========================================
# 5. 训练函数 
# ==========================================
def train_model(csv_path, prefix="", res=128):
    seed_everything(CONFIG["SEED"])
    full_df = pd.read_csv(csv_path)
    
    model_type = "CRNN"
    model_key = f"{model_type}_{res}{prefix}"
    
    print(f"\n{'='*50}")
    print(f"Processing: {model_key} (Res: {res})")
    print(f"{'='*50}")
    
    scores = []

    for fold_id in range(1, 9):
        # 目标保存路径（Working 目录）
        save_name = SAVE_DIR / f"{model_key}_fold_{fold_id}.pth"
        external_source = EXTERNAL_WEIGHTS_DIR / save_name.name
        
        # 如果 Working 里没有，但 Input 里有，先拷贝过来！
        if (not save_name.exists()) and external_source.exists():
            print(f"    Found pretrained weight in Input: {external_source.name}")
            try:
                shutil.copy(external_source, save_name)
                print(f"    Copied to working dir. Training will be skipped.")
            except Exception as e:
                print(f"    Copy failed: {e}")

        run_inference = False
        needs_training = True

        if save_name.exists():
            print(f"    Fold {fold_id}: Found existing .pth. Checking...", end="\r")
            run_inference = True
            needs_training = False 
        
        model = AdvancedCRNN(input_mels=res).to(CONFIG["DEVICE"])

        if run_inference:
            try:
                state_dict = torch.load(save_name, map_location=CONFIG["DEVICE"])
                new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict)
            except:
                print(f"\n    old {fold_id}: File corrupt. Retraining...")
                needs_training = True
                run_inference = False

        if needs_training:
            print(f"  Fold {fold_id}: Training start...")
            train_df = full_df[full_df['fold'] != fold_id].reset_index(drop=True)
            val_df = full_df[full_df['fold'] == fold_id].reset_index(drop=True)
            
            train_ds = UrbanSoundDataset(train_df, NPY_DIR, 'train', res, cache_data=True)
            train_loader = DataLoader(train_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=CONFIG["NUM_WORKERS"], pin_memory=True, persistent_workers=True)
            val_loader = DataLoader(UrbanSoundDataset(val_df, NPY_DIR, 'train', res, cache_data=True), batch_size=CONFIG["BATCH_SIZE"]*2, shuffle=False, num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)

            try: model = torch.compile(model) 
            except: pass

            scaler = torch.amp.GradScaler('cuda')
            criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)
            optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LR"], weight_decay=5e-2)
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
                    with torch.amp.autocast('cuda'):
                        imgs, la, lb, lam = Augmenter.get_batch(imgs, lbls)
                        out = model(imgs)
                        loss = lam * criterion(out, la) + (1-lam) * criterion(out, lb)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})

                if (epoch < 10) or (epoch % 2 != 0 and epoch < 35): continue

                model.eval()
                preds, targs = [], []
                with torch.no_grad():
                    for imgs, lbls in val_loader:
                        imgs = imgs.to(CONFIG["DEVICE"])
                        with torch.amp.autocast('cuda'):
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
            
            state_dict = torch.load(save_name, map_location=CONFIG["DEVICE"])
            new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            try: model.load_state_dict(new_state_dict)
            except: model.load_state_dict(state_dict)

        if not needs_training:
            val_df = full_df[full_df['fold'] == fold_id].reset_index(drop=True)
            val_loader = DataLoader(UrbanSoundDataset(val_df, NPY_DIR, 'train', res, cache_data=True), batch_size=CONFIG["BATCH_SIZE"]*2, shuffle=False, num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)

        model.eval()
        preds, targs = [], []
        with torch.no_grad():
            for imgs, lbls in tqdm(val_loader, desc=f"Eval F{fold_id}", leave=False):
                imgs = imgs.to(CONFIG["DEVICE"])
                with torch.amp.autocast('cuda'):
                    out = model(imgs)
                preds.extend(out.argmax(1).cpu().numpy())
                targs.extend(lbls.numpy())
        
        fold_acc = accuracy_score(targs, preds)
        scores.append(fold_acc)
        print(f"   [Fold {fold_id}] Accuracy: {fold_acc:.4f}")
        
        del model; gc.collect(); torch.cuda.empty_cache()

    avg_score = np.mean(scores)
    print(f"{model_key} Summary: Avg={avg_score:.4f}")
    return avg_score

# ==========================================
# 6. 推理函数 
# ==========================================
def predict_with_tta(model, imgs):
    outputs = []
    
    with torch.amp.autocast('cuda'):
        base_out = torch.softmax(model(imgs), dim=1)
        outputs.append((base_out, 1.5))
    
    for shift in CONFIG["TTA_SHIFTS"]:
        if shift == 0: continue
        aug = torch.roll(imgs, shifts=shift, dims=3)
        with torch.amp.autocast('cuda'):
            outputs.append((torch.softmax(model(aug), dim=1), 1.1))
            
    for shift in [-2, 2]:
        aug = torch.roll(imgs, shifts=shift, dims=2)
        with torch.amp.autocast('cuda'):
            outputs.append((torch.softmax(model(aug), dim=1), 0.9))
    
    for gain in [0.9, 1.1]:
        aug = imgs * gain
        with torch.amp.autocast('cuda'):
             outputs.append((torch.softmax(model(aug), dim=1), 1.0))

    final_prob = 0
    total_weight = 0
    for prob, w in outputs:
        final_prob += prob * w
        total_weight += w
        
    return final_prob / total_weight

def inference(prefix="", res=128):
    print(f"\nInference [{prefix}] Res:{res}...")
    test_df = pd.read_csv(TEST_CSV_PATH)
    final_probs = torch.zeros((len(test_df), 10), device=CONFIG["DEVICE"])
    cnt = 0
    
    model_key = f"CRNN_{res}{prefix}"
    ds = UrbanSoundDataset(test_df, NPY_DIR, 'test', res, cache_data=True)
    loader = DataLoader(ds, batch_size=CONFIG["BATCH_SIZE"]*2, shuffle=False, num_workers=2)

    for fold in range(1, 9):
        path = SAVE_DIR / f"{model_key}_fold_{fold}.pth"
        if not path.exists(): continue
        
        model = AdvancedCRNN(input_mels=res).to(CONFIG["DEVICE"])
        try:
            state_dict = torch.load(path, map_location=CONFIG["DEVICE"])
            new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
        except: continue
        model.eval()

        fold_probs = []
        with torch.no_grad():
            for imgs, _, _ in tqdm(loader, desc=f"F{fold}", leave=False):
                imgs = imgs.to(CONFIG["DEVICE"])
                fold_probs.append(predict_with_tta(model, imgs))
        
        final_probs += torch.cat(fold_probs)
        cnt += 1
        del model; gc.collect()

    if cnt > 0: final_probs /= cnt
    return final_probs.cpu()


# ==========================================
# 7. 主流程执行
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("STARTING DENSE MULTI-RESOLUTION ENSEMBLE")
    print("   Logic: Check INPUT dir first. If exists -> Copy & Skip Train.")
    print("="*60)
    SOURCE_DIRS = [
        Path('/kaggle/input/pth-final')
    ]
    
    print("Moving Pretrained Weights to Working Directory...")
    for source_dir in SOURCE_DIRS:
        if not source_dir.exists():
            print(f"   Source dir not found: {source_dir}, skipping.")
            continue
            
        # 遍历该目录下的所有 .pth 文件
        for pth_file in source_dir.glob("*.pth"):
            dest_path = SAVE_DIR / pth_file.name
            
            # 只有当目标文件不存在时才复制 (避免重复拷贝浪费时间)
            if not dest_path.exists():
                try:
                    shutil.copy(pth_file, dest_path)
                    print(f"   Copied: {pth_file.name}")
                except Exception as e:
                    print(f"   Copy Failed: {pth_file.name} | Error: {e}")
            else:
                print(f"   Exists: {pth_file.name}")
    
    print("="*60 + "\n")

    # 1. 设置 5 路分辨率
    CONFIG["FINAL_RESOLUTIONS"] = [128, 160, 192, 256, 320] 
    # 2. 初始化
    final_ensemble_probs = 0
    # 3. 定义权重 (密集集成策略)
    res_weights = {
        128: 0.6,   # 降权：它主要用来防止过拟合，但准确度不如大图
        160: 0.9,   
        192: 1.3,   
        256: 1.5,   
        320: 1.2    # 保持高位
    }
    total_weight = sum(res_weights.values())

    for res in CONFIG["FINAL_RESOLUTIONS"]:
        current_bs = 64
        if res >= 320: current_bs = 48
        elif res >= 256: current_bs = 56
        CONFIG["BATCH_SIZE"] = current_bs
        print(f"\nResolution: {res} | Batch Size: {current_bs}")
        # A. 训练 (如果文件在 Input 里有，会自动拷贝并跳过)
        # 注意：这里 prefix 保持为 "_final_pseudo"，确保文件名匹配
        train_model(TRAIN_CSV_PATH, prefix="_final_pseudo", res=res) 
        # B. 推理
        preds = inference(prefix="_final_pseudo", res=res)
        # C. 加权集成
        w = res_weights.get(res, 1.0)
        print(f"    Merging Resolution {res} with weight {w}")
        final_ensemble_probs += (preds.to(CONFIG["DEVICE"]) * w)

    # 归一化
    final_ensemble_probs /= total_weight
    # Power Sharpening (锐化)
    # 系数 1.05 - 1.10 比较安全。太大会导致甚至错判也被放大。
    print("Applying Power Sharpening (Power=1.05)...")
    final_ensemble_probs = final_ensemble_probs ** 1.05
    # 重新归一化一下（虽然argmax不需要，但为了后续乘权重数学上严谨）
    final_ensemble_probs = final_ensemble_probs / final_ensemble_probs.sum(dim=1, keepdim=True)
    # 先应用基础权重
    # 针对常见混淆的强力修正权重
    class_weights = torch.tensor([1.85, 1.00, 0.85, 0.95, 1.25, 1.10, 1.00, 0.65, 1.35, 0.55]).to(CONFIG['DEVICE'])
    probs = final_ensemble_probs * class_weights
    conflict_mask_5_0 = (torch.argmax(probs, dim=1) == 5)
    condition_5_0 = probs[:, 0] > (probs[:, 5] * 0.60) 
    probs[conflict_mask_5_0 & condition_5_0, 0] *= 1.5 
    conflict_mask_9 = (torch.argmax(probs, dim=1) == 9) 
    condition_9_2 = probs[:, 2] > (probs[:, 9] * 0.70)
    probs[conflict_mask_9 & condition_9_2, 2] *= 1.5
    condition_9_3 = probs[:, 3] > (probs[:, 9] * 0.70)
    probs[conflict_mask_9 & condition_9_3, 3] *= 1.5
    
    
    conflict_mask_7_4 = (torch.argmax(probs, dim=1) == 7) 
    condition_7_4 = probs[:, 4] > (probs[:, 7] * 0.75)
    probs[conflict_mask_7_4 & condition_7_4, 4] *= 1.4
    
    final_ensemble_probs = probs
    # -------------------------------------
    # Step 3: 生成提交
    # -------------------------------------
    print("Saving Submission...")
    final_preds = torch.argmax(final_ensemble_probs, dim=1).cpu().numpy()
    
    test_df = pd.read_csv(TEST_CSV_PATH)
    sub = pd.DataFrame({'ID': range(len(test_df)), 'TARGET': final_preds})
    sub.to_csv("submission.csv", index=False)
    
    print("\nDONE! Dense Ensemble Submission Generated.")