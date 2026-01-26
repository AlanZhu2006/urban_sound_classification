import os
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm

# ================= 配置 =================
# 使用你提供的路径
BASE_DIR = Path("C://Users//zhuya//Desktop//Kaggle_Data")
# =======================================

TRAIN_CSV = BASE_DIR / "metadata" / "kaggle_train.csv"
TEST_CSV = BASE_DIR / "metadata" / "kaggle_test.csv"
AUDIO_DIR = BASE_DIR / "audio"
SAVE_DIR = BASE_DIR / "processed_npy" # 缓存保存的路径

CONFIG = {
    "SR": 22050,
    "N_MELS": 128,
    "DURATION": 4.0,
    "MAX_LEN": 173
}

def process_and_save(df, subset_name):
    print(f"Processing {subset_name} data...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['slice_file_name']
        
        # 1. 确定源文件路径
        if 'fold' in row: # 训练集
            audio_path = AUDIO_DIR / f"fold{row['fold']}" / filename
            save_folder = SAVE_DIR / f"fold{row['fold']}"
        else: # 测试集
            audio_path = AUDIO_DIR / "test" / filename
            save_folder = SAVE_DIR / "test"
            
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = save_folder / (filename + ".npy")
        
        if save_path.exists(): continue

        try:
            # 2. 加载与转换
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

            current_width = mel_spec_db.shape[1]
            if current_width < CONFIG["MAX_LEN"]:
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, CONFIG["MAX_LEN"] - current_width)), mode='constant')
            else:
                mel_spec_db = mel_spec_db[:, :CONFIG["MAX_LEN"]]
            
            # 3. 保存
            np.save(save_path, mel_spec_db.astype(np.float32))
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    if not BASE_DIR.exists():
        print(f"❌ 错误: 路径不存在 {BASE_DIR}")
    else:
        train_df = pd.read_csv(TRAIN_CSV)
        process_and_save(train_df, "Train")
        
        test_df = pd.read_csv(TEST_CSV)
        process_and_save(test_df, "Test")
        
        print(f"\n✅ 预处理完成！缓存已保存在: {SAVE_DIR}")