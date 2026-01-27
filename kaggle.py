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