# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
--------------------------------------------------------------------------------------------------------
# ==========================================
# 单元格 1: Kaggle 环境输入数据加载
# ==========================================
import os
from pathlib import Path

# 1. Kaggle Dataset 根目录（自动挂载，无需你做任何操作）
# 你只需要在左侧 Add Data 中添加你的数据集
BASE_DIR = Path("/kaggle/input/kaggle-data/Kaggle_Data")  
# ↑ 把 kaggle-data 换成你 Dataset 的名字（不要空格）

print(f"📂 数据目录: {BASE_DIR}")

# 2. 自动检查目录结构
required_files = [
    BASE_DIR / "metadata" / "kaggle_train.csv",
    BASE_DIR / "metadata" / "kaggle_test.csv",
]

missing = False
for f in required_files:
    if not f.exists():
        print(f"❌ 未找到文件: {f}")
        missing = True

if not missing:
    print("✅ 数据结构检查通过！")

# 3. 自动补偿路径（如果压缩包外面多包了一层文件夹）
subdirs = list(BASE_DIR.iterdir())

if len(subdirs) == 1 and subdirs[0].is_dir():
    print(f"🔄 检测到包裹目录：{subdirs[0].name}，自动进入其中")
    BASE_DIR = subdirs[0]

print(f"📌 最终数据根目录: {BASE_DIR}")



# Parallel audio processing pipeline that converts waveforms to Mel-spectrograms, saves them as .npy files, and zips the results for download.

import os
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed  # 用于并行处理

# ================= 配置区域 =================
CONFIG = {
    "SR": 22050,
    "N_MELS": 128,
    "DURATION": 4.0,
    "MAX_LEN": 173
}
import shutil

----------------------------------------------------------------------------------------

# ==========================================

# 1. 修复路径空格问题
BASE_DIR = Path("/kaggle/input/kaggle-data/Kaggle_Data") 

TRAIN_CSV = BASE_DIR / "metadata" / "kaggle_train.csv"
TEST_CSV = BASE_DIR / "metadata" / "kaggle_test.csv"
AUDIO_DIR = BASE_DIR / "audio"

SAVE_DIR = Path("/kaggle/working/processed_npy")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print(f"检测输入路径: {BASE_DIR}")
if not BASE_DIR.exists():
    print("❌ 错误: 找不到输入目录，请检查 Dataset 是否正确挂载！")
# ===========================================

def process_one_file(row, subset_name):
    """
    处理单个文件的函数，用于并行调用
    """
    try:
        filename = str(row['slice_file_name'])
        # 确保文件名以 .wav 结尾 (视 CSV 内容而定)
        if not filename.endswith('.wav'):
            filename += '.wav'

        # 构建保存路径
        if 'fold' in row:
            fold_dir = f"fold{row['fold']}"
            audio_path = AUDIO_DIR / fold_dir / filename
            save_folder = SAVE_DIR / fold_dir
        else:
            audio_path = AUDIO_DIR / "test" / filename
            save_folder = SAVE_DIR / "test"

        # 确保子文件夹存在 (多线程下 makedirs 需要 exist_ok=True)
        save_folder.mkdir(parents=True, exist_ok=True)
        
        save_path = save_folder / (filename.replace('.wav', '.npy'))

        if save_path.exists():
            return "Skipped"

        # Librosa 加载与处理
        y, _ = librosa.load(str(audio_path), sr=CONFIG["SR"], duration=CONFIG["DURATION"])

        target_len = int(CONFIG["SR"] * CONFIG["DURATION"])
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')
        else:
            y = y[:target_len]

        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=CONFIG["SR"], n_mels=CONFIG["N_MELS"],
            fmax=8000, hop_length=512
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 统一宽度
        current_width = mel_spec_db.shape[1]
        if current_width < CONFIG["MAX_LEN"]:
            mel_spec_db = np.pad(
                mel_spec_db, ((0,0),(0, CONFIG["MAX_LEN"] - current_width)),
                mode='constant'
            )
        else:
            mel_spec_db = mel_spec_db[:, :CONFIG["MAX_LEN"]]

        np.save(save_path, mel_spec_db.astype(np.float32))
        return "Success"

    except Exception as e:
        return f"Error: {filename} - {e}"

def run_processing(df, name):
    print(f"🚀 开始并行处理 {name} ({len(df)} 个文件)...")
    
    # n_jobs=-1 使用所有 CPU 核心，backend="multiprocessing" 适合计算密集型
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_one_file)(row, name) for idx, row in tqdm(df.iterrows(), total=len(df))
    )
    
    # 简单的错误统计
    errors = [r for r in results if r.startswith("Error")]
    print(f"✅ {name} 完成。成功: {len(results) - len(errors)}, 失败: {len(errors)}")
    if errors:
        print("前5个错误样例:", errors[:5])

# 读取 CSV
if TRAIN_CSV.exists() and TEST_CSV.exists():
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    run_processing(train_df, "Train")
    run_processing(test_df, "Test")
    
    print("\n📦 正在打包数据以方便下载...")
    import shutil
    shutil.make_archive('/kaggle/working/processed_data', 'zip', '/kaggle/working/processed_npy')
    print("🎉 全部完成！请下载 processed_data.zip")
else:
    print(f"❌ 找不到 CSV 文件，请检查路径: {TRAIN_CSV}")


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
    # 如果模型在 A 和 B 之间犹豫，且分差很小，强制判给“弱势群体”。
    # ---------------------------------------------------------
    # 修正 A: Engine (5) vs Air_conditioner (0)
    # ---------------------------------------------------------
    conflict_mask_5_0 = (torch.argmax(probs, dim=1) == 5)
    
    # ⬆️ 回调：从 0.50 提回 0.60。
    condition_5_0 = probs[:, 0] > (probs[:, 5] * 0.60) 
    
    # ⬇️ 降力度：从 2.0 降回 1.5。
    probs[conflict_mask_5_0 & condition_5_0, 0] *= 1.5 
    
    # ---------------------------------------------------------
    # 修正 B: Street_music (9) vs Others
    # ---------------------------------------------------------
    conflict_mask_9 = (torch.argmax(probs, dim=1) == 9) 
    
    # B1: Music vs Children
    condition_9_2 = probs[:, 2] > (probs[:, 9] * 0.70)
    probs[conflict_mask_9 & condition_9_2, 2] *= 1.5

    # B2: Music vs Dog
    condition_9_3 = probs[:, 3] > (probs[:, 9] * 0.70)
    probs[conflict_mask_9 & condition_9_3, 3] *= 1.5
    
    # ---------------------------------------------------------
    # 修正 C: Jackhammer (7) vs Drilling (4)
    # ---------------------------------------------------------
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




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
import os

# ==========================================
# 1. 关键路径配置 (请务必核对)
# ==========================================
# A. 预测结果文件 (生成的 submission.csv)
PRED_FILE = Path("/kaggle/input/1234567/submission (22).csv")
DATASET_DIR = Path('/kaggle/input/kaggle-data/Kaggle_Data') 
TEST_META_FILE = DATASET_DIR / "metadata" / "kaggle_test.csv"
GT_FILE = Path('/kaggle/input/test-final/test.csv')

# ==========================================
# 2. 检查文件与加载
# ==========================================
if not PRED_FILE.exists():
    print(f"❌ 错误: 找不到预测文件: {PRED_FILE}")
elif not TEST_META_FILE.exists():
    print(f"❌ 错误: 找不到测试集元数据: {TEST_META_FILE}")
elif not GT_FILE.exists():
    print(f"❌ 错误: 找不到真实标签文件: {GT_FILE}")
else:
    try:
        # 读取数据
        preds_df = pd.read_csv(PRED_FILE)
        test_meta_df = pd.read_csv(TEST_META_FILE)
        ground_truth_df = pd.read_csv(GT_FILE)

        print("✅ 文件加载成功，正在对齐数据...")

        # -------------------------------------------------
        # 步骤 A: 对齐数据 (ID -> 文件名 -> 真实标签)
        # -------------------------------------------------
        if 'ID' not in preds_df.columns:
            preds_df = preds_df.reset_index().rename(columns={'index': 'ID'})
        
        preds_df = preds_df.rename(columns={'TARGET': 'pred_classID'})
        
        # 1. 把预测结果的 ID 换成 文件名
        merged_step1 = preds_df.merge(test_meta_df[['ID', 'slice_file_name']], on='ID', how='left')

        # 2. 准备真实标签 (只要文件名和类别)
        gt_clean = ground_truth_df[['slice_file_name', 'classID']].rename(columns={'classID': 'true_classID'})
        
        # 3. 最终合并
        final_df = merged_step1.merge(gt_clean, on='slice_file_name', how='inner')

        # -------------------------------------------------
        # 步骤 B: 计算指标
        # -------------------------------------------------
        y_true = final_df['true_classID'].values
        y_pred = final_df['pred_classID'].values
        
        # 类别名称
        class_names = [
            "0", "1", "2", "3", 
            "4", "5", "6", "7", 
            "8", "9"
        ]

        # 1. 准确率
        acc = accuracy_score(y_true, y_pred)
        print("\n" + "═"*40)
        print(f"🏆 本地验证准确率 (Accuracy): {acc:.5f}")
        print("═"*40)

        # 2. 分类报告
        print("\n📊 Classification Report:\n")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

        # 3. 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred) # 计算混淆矩阵
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix')
        plt.tight_layout()
        plt.show() # 

        # -------------------------------------------------
        # 步骤 C: 自动计算建议权重 (修正版)
        # -------------------------------------------------
        print("\n⚖️ 根据混淆矩阵计算建议权重 (Auto-Weighting):")
        
        # row_sums = 真实标签的数量 (Ground Truth Distribution)
        row_sums = cm.sum(axis=1)
        # col_sums = 你预测出的数量 (Predicted Distribution)
        col_sums = cm.sum(axis=0)

        # 打印分布对比
        print(f"{'Class':<20} | {'True Count':<10} | {'Pred Count':<10} | {'Status'}")
        print("-" * 60)
        for i, name in enumerate(class_names):
            diff = col_sums[i] - row_sums[i]
            status = "OK"
            if diff > row_sums[i] * 0.1: status = "Over-predicted (需降权)"
            elif diff < -row_sums[i] * 0.1: status = "Under-predicted (需加权)"
            print(f"{name:<20} | {row_sums[i]:<10} | {col_sums[i]:<10} | {status}")

        # 计算权重公式： 真实数量 / (预测数量 + 极小值)
        raw_weights = row_sums / (col_sums + 1e-6)
        
        # 归一化 (让均值为 1)
        suggested_weights = raw_weights / raw_weights.mean()

        print("\n💡 建议的权重列表 (可直接复制到 Inference 代码):")
        print("-" * 50)
        # 格式化输出，方便复制
        w_str = ", ".join([f"{w:.2f}" for w in suggested_weights])
        print(f"class_weights = torch.tensor([{w_str}]).to(CONFIG['DEVICE'])")
        print("-" * 50)

    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()


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

# 输出路径 (Working)
SAVE_DIR = Path('/kaggle/working/models')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. 终极 CONFIG
# ==========================================
CONFIG = {
    # 🌟 异构集成：两种架构同时跑
    "MODELS": ["CRNN", "EffNet"], 
    "FINAL_RESOLUTIONS": [128, 96, 64], 
    
    "BATCH_SIZE": 64,             
    "EPOCHS": 50,                 
    "PATIENCE": 12,               
    "LR": 1e-3,
    "MIN_LR": 1e-6,
    "SEED": 2024,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    
    # ⚡ 增强策略
    "TTA_SHIFTS": [0, -4, 4],     # Pixel shift
    "USE_CUTMIX": True,
    "CUTMIX_PROB": 0.4,
    "MIXUP_PROB": 0.3,            
    "NOISE_LEVEL": 0.015,         # ✅ 高斯噪声等级
    
    # 🧗 Hill Climbing 设置
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

        # 🔥 核心：显式加载数据并显示进度条
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
        # ✅ 新增：高斯噪声
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
    # ✅ 核心：Generalized Mean Pooling
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
        
        # ✅ 使用 GeM 替代 AdaptiveAvgPool
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
        self.gem = GeM() # ✅ 这里也用 GeM
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
    
    # ✅ 修复: 初始化返回值变量
    oof_results = {'true': [], 'pred': []}
    
    # 清空分数板
    if prefix != "": MODEL_SCORES.clear()

    for model_type in CONFIG["MODELS"]:
        for res in CONFIG["FINAL_RESOLUTIONS"]:
            model_key = f"{model_type}_{res}{prefix}"
            print(f"\n🚀 Processing: {model_key} (Res: {res})")

            for fold_id in range(1, 9):
                save_name = SAVE_DIR / f"{model_key}_fold_{fold_id}.pth"
                
                # =================================================
                # 🟢 关键逻辑：检查 Working 目录中是否已有模型
                # =================================================
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
                        print(f"   ✅ Fold {fold_id}: Loaded successfully. Skipping Training.")
                    except:
                        print(f"\n   ⚠️ Fold {fold_id}: Model architecture mismatch or corrupt. Retraining...")
                        needs_training = True # Load Failed, force retrain
                        run_inference = False

                # === Training Flow ===
                if needs_training:
                    print(f"   🔥 Fold {fold_id}: Training start...")
                    val_df = full_df[full_df['fold'] == fold_id].reset_index(drop=True)
                    train_df = full_df[full_df['fold'] != fold_id].reset_index(drop=True)
                    
                    # Dataset & Loader
                    train_ds = UrbanSoundDataset(train_df, NPY_DIR, 'train', res, cache_data=True)
                    train_loader = DataLoader(train_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, 
                                            num_workers=CONFIG["NUM_WORKERS"], pin_memory=True, persistent_workers=True)
                    
                    val_ds = UrbanSoundDataset(val_df, NPY_DIR, 'train', res, cache_data=True)
                    val_loader = DataLoader(val_ds, batch_size=CONFIG["BATCH_SIZE"]*2, shuffle=False, 
                                          num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)

                    print(f"   ⏳ [Status] Compiling model (One-time setup)...")
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
    print("\n🧗 Running Hill Climbing Optimization (Heuristic)...")
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
    print(f"\n🔮 Ensemble Inference [{prefix}]...")
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
# 9. 主流程 (安全合规版)
# ==========================================
if __name__ == "__main__":
    
    seed_everything(CONFIG["SEED"])
    PRETRAINED_SOURCE_DIR = Path('/kaggle/input/train-pth') 
    
    print(f"🔍 Checking for pretrained models in: {PRETRAINED_SOURCE_DIR}")
    
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
                print(f"   -> 🚚 Copied to Working: {file_path.name}")
            else:
                print(f"   -> ⏩ Already exists: {file_path.name}")
    else:
        print("⚠️ Pretrained directory not found. Please check the path name!")
    
    # 检查 Working 目录下的文件情况
    print("📂 Checking /kaggle/working/models ...")
    files = list(SAVE_DIR.glob("*.pth"))
    if len(files) > 0:
        print(f"✅ Found {len(files)} models in Working directory. Will use them.")
    else:
        print("⚠️ Working directory empty. Starting fresh training...")

    expected_files = []
    for m in ["CRNN", "EffNet"]:
        for r in [128, 96, 64]:
            for f in range(1, 9):
                expected_files.append(f"{m}_{r}_fold_{f}.pth")
    
    existing_files = [f.name for f in SAVE_DIR.glob("*.pth")]
    missing = set(expected_files) - set(existing_files)
    
    print(f"缺少 {len(missing)} 个文件:")
    for f in missing:
        print(f"❌ {f}")

    # 1. 自动检测 Working 里的模型 (有则跳过，无则训练)
    oof_data = train_multires(TRAIN_CSV_PATH, prefix="")
    
    # 2. 计算权重 (使用 OOF 分数进行爬山算法优化)
    best_weights = hill_climbing_optimize()
    
    # 3. 预测 (生成测试集概率)
    probs = ensemble_inference(prefix="", optimal_weights=best_weights)
    
    # 4. 伪标签流程 (如果开启)
    if CONFIG["ENABLE_PSEUDO"]:
        print("\n🔮 Starting Pseudo Labeling...")
        max_p, preds = torch.max(probs, 1)
        test_df = pd.read_csv(TEST_CSV_PATH)
        
        # 只选择置信度 > 0.90 的样本
        mask = max_p > 0.90
        pseudo_df = test_df[mask.numpy()].copy()
        print(f"🔥 Selected {len(pseudo_df)} pseudo samples.")
        
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
            print("✅ Final Submission Saved (with Pseudo Labeling)!")
        else:
            print("⚠️ Not enough pseudo labels found. Using original predictions.")
            sub = pd.DataFrame({'ID': range(len(test_df)), 'TARGET': probs.argmax(1).numpy()})
            sub.to_csv("submission.csv", index=False)
            print("✅ Submission Saved!")
    else:
        # 如果不开启伪标签，直接保存第一轮结果
        sub = pd.DataFrame({'ID': range(len(test_df)), 'TARGET': probs.argmax(1).numpy()})
        sub.to_csv("submission.csv", index=False)
        print("✅ Submission Saved!")