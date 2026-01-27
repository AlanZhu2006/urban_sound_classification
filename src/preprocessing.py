"""
数据预处理模块
将原始音频文件转换为 Mel 频谱图并保存为 .npy 文件
"""
import os
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import shutil

from .config import AUDIO_CONFIG, BASE_DIR, NPY_DIR, TRAIN_CSV_PATH, TEST_CSV_PATH


def process_single_audio(row: dict, audio_dir: Path, save_dir: Path, subset_name: str) -> str:
    """
    处理单个音频文件，转换为 Mel 频谱图
    
    Args:
        row: DataFrame 行数据
        audio_dir: 音频文件目录
        save_dir: 保存目录
        subset_name: 子集名称 (train/test)
    
    Returns:
        处理状态字符串
    """
    try:
        filename = str(row['slice_file_name'])
        if not filename.endswith('.wav'):
            filename += '.wav'
        
        # 构建路径
        if 'fold' in row:
            fold_dir = f"fold{row['fold']}"
            audio_path = audio_dir / fold_dir / filename
            save_folder = save_dir / fold_dir
        else:
            audio_path = audio_dir / "test" / filename
            save_folder = save_dir / "test"
        
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = save_folder / (filename + '.npy')
        
        # 跳过已存在的文件
        if save_path.exists():
            return "Skipped"
        
        # 加载音频
        y, _ = librosa.load(str(audio_path), sr=AUDIO_CONFIG["SR"], 
                           duration=AUDIO_CONFIG["DURATION"])
        
        # 填充或截断
        target_len = int(AUDIO_CONFIG["SR"] * AUDIO_CONFIG["DURATION"])
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')
        else:
            y = y[:target_len]
        
        # 计算 Mel 频谱图
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=AUDIO_CONFIG["SR"], 
            n_mels=AUDIO_CONFIG["N_MELS"],
            fmax=AUDIO_CONFIG["FMAX"], 
            hop_length=AUDIO_CONFIG["HOP_LENGTH"]
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 统一宽度
        current_width = mel_spec_db.shape[1]
        if current_width < AUDIO_CONFIG["MAX_LEN"]:
            mel_spec_db = np.pad(
                mel_spec_db, 
                ((0, 0), (0, AUDIO_CONFIG["MAX_LEN"] - current_width)),
                mode='constant'
            )
        else:
            mel_spec_db = mel_spec_db[:, :AUDIO_CONFIG["MAX_LEN"]]
        
        # 保存
        np.save(save_path, mel_spec_db.astype(np.float32))
        return "Success"
    
    except Exception as e:
        return f"Error: {filename} - {e}"


def run_preprocessing(df: pd.DataFrame, audio_dir: Path, save_dir: Path, 
                     name: str, n_jobs: int = -1) -> None:
    """
    并行处理数据集中的所有音频文件
    
    Args:
        df: 包含文件信息的 DataFrame
        audio_dir: 音频目录
        save_dir: 保存目录
        name: 数据集名称
        n_jobs: 并行任务数 (-1 表示使用所有 CPU)
    """
    print(f"🚀 开始并行处理 {name} ({len(df)} 个文件)...")
    
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_single_audio)(row, audio_dir, save_dir, name) 
        for _, row in tqdm(df.iterrows(), total=len(df))
    )
    
    # 统计结果
    success = sum(1 for r in results if r == "Success")
    skipped = sum(1 for r in results if r == "Skipped")
    errors = [r for r in results if r.startswith("Error")]
    
    print(f"✅ {name} 完成。成功: {success}, 跳过: {skipped}, 失败: {len(errors)}")
    if errors:
        print("前5个错误样例:", errors[:5])


def preprocess_all(audio_dir: Path = None, save_dir: Path = None) -> None:
    """
    预处理所有训练集和测试集音频
    
    Args:
        audio_dir: 音频目录，默认使用配置中的路径
        save_dir: 保存目录，默认使用配置中的路径
    """
    if audio_dir is None:
        audio_dir = BASE_DIR / "audio"
    if save_dir is None:
        save_dir = NPY_DIR
    
    # 检查 CSV 文件
    if not TRAIN_CSV_PATH.exists():
        raise FileNotFoundError(f"找不到训练集 CSV: {TRAIN_CSV_PATH}")
    if not TEST_CSV_PATH.exists():
        raise FileNotFoundError(f"找不到测试集 CSV: {TEST_CSV_PATH}")
    
    # 读取数据
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)
    
    # 处理
    run_preprocessing(train_df, audio_dir, save_dir, "Train")
    run_preprocessing(test_df, audio_dir, save_dir, "Test")
    
    print("\n🎉 所有音频预处理完成!")


def create_zip_archive(source_dir: Path, output_path: Path) -> None:
    """
    将处理后的数据打包为 ZIP
    
    Args:
        source_dir: 源目录
        output_path: 输出路径 (不含 .zip 扩展名)
    """
    print("📦 正在打包数据...")
    shutil.make_archive(str(output_path), 'zip', str(source_dir))
    print(f"✅ 打包完成: {output_path}.zip")


if __name__ == "__main__":
    preprocess_all()
