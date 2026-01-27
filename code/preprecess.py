# Parallel audio processing pipeline that converts waveforms to Mel-spectrograms, saves them as .npy files, and zips the results for download.
# Important: this script has to be run in a Kaggle Notebook environment where the dataset is mounted!!!
import os
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed  # For parallel processing
import shutil

# ================= Configuration Area =================
CONFIG = {
    "SR": 22050,       # Sampling Rate
    "N_MELS": 128,     # Number of Mel bands (Frequency resolution)
    "DURATION": 4.0,   # Duration in seconds
    "MAX_LEN": 173     # Fixed time dimension for the output tensor
}

# ======================================================

# 1. Fix path spacing issues
BASE_DIR = Path("/kaggle/input/kaggle-data/Kaggle_Data")

TRAIN_CSV = BASE_DIR / "metadata" / "kaggle_train.csv"
TEST_CSV = BASE_DIR / "metadata" / "kaggle_test.csv"
AUDIO_DIR = BASE_DIR / "audio"

SAVE_DIR = Path("/kaggle/working/processed_npy")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Checking input path: {BASE_DIR}")
if not BASE_DIR.exists():
    print("❌ Error: Input directory not found. Please check if the Dataset is mounted correctly!")

# ======================================================

def process_one_file(row, subset_name):
    """
    Function to process a single file, used for parallel execution.
    Converts audio to Log-Mel Spectrogram and saves as .npy.
    """
    try:
        filename = str(row['slice_file_name'])
        # Ensure filename ends with .wav (depends on CSV content)
        if not filename.endswith('.wav'):
            filename += '.wav'

        # Construct save path
        if 'fold' in row:
            fold_dir = f"fold{row['fold']}"
            audio_path = AUDIO_DIR / fold_dir / filename
            save_folder = SAVE_DIR / fold_dir
        else:
            audio_path = AUDIO_DIR / "test" / filename
            save_folder = SAVE_DIR / "test"

        # Ensure subfolder exists (makedirs requires exist_ok=True for multi-threading)
        save_folder.mkdir(parents=True, exist_ok=True)
        
        # Define the output .npy filename
        save_path = save_folder / (filename.replace('.wav', '.npy'))

        # Skip if already exists
        if save_path.exists():
            return "Skipped"

        # Librosa loading and processing
        # 1. Load audio
        y, _ = librosa.load(str(audio_path), sr=CONFIG["SR"], duration=CONFIG["DURATION"])

        # 2. Pad or truncate to fixed duration
        target_len = int(CONFIG["SR"] * CONFIG["DURATION"])
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')
        else:
            y = y[:target_len]

        # 3. Extract Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=CONFIG["SR"], n_mels=CONFIG["N_MELS"],
            fmax=8000, hop_length=512
        )
        
        # 4. Convert to Log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 5. Unify width (Padding or Cropping time dimension)
        current_width = mel_spec_db.shape[1]
        if current_width < CONFIG["MAX_LEN"]:
            mel_spec_db = np.pad(
                mel_spec_db, ((0,0),(0, CONFIG["MAX_LEN"] - current_width)),
                mode='constant'
            )
        else:
            mel_spec_db = mel_spec_db[:, :CONFIG["MAX_LEN"]]

        # 6. Save as float32 .npy file
        np.save(save_path, mel_spec_db.astype(np.float32))
        return "Success"

    except Exception as e:
        return f"Error: {filename} - {e}"

def run_processing(df, name):
    print(f"🚀 Starting parallel processing for {name} ({len(df)} files)...")
    
    # n_jobs=-1 uses all CPU cores, backend="loky" is suitable for robust parallel execution
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_one_file)(row, name) for idx, row in tqdm(df.iterrows(), total=len(df))
    )
    
    # Simple error statistics
    errors = [r for r in results if r.startswith("Error")]
    print(f"✅ {name} completed. Success: {len(results) - len(errors)}, Failed: {len(errors)}")
    if errors:
        print("Top 5 error examples:", errors[:5])

# Main Execution Flow
if TRAIN_CSV.exists() and TEST_CSV.exists():
    # Read CSVs
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # Run processing
    run_processing(train_df, "Train")
    run_processing(test_df, "Test")
    
    # Zip the processed data
    print("\n📦 Zipping data for easy download...")
    shutil.make_archive('/kaggle/working/processed_data', 'zip', '/kaggle/working/processed_npy')
    print("🎉 All done! Please download processed_data.zip")
else:
    print(f"❌ CSV file not found, please check path: {TRAIN_CSV}")