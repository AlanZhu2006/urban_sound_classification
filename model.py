"""
One piece of audio = One piece of image
"""

import pandas as pd
import librosa
import numpy as np
from pathlib import Path

# Data paths
BASE_DIR = Path("Your Path")
TRAIN_CSV = BASE_DIR / "metadata" / "kaggle_train.csv"
AUDIO_DIR = BASE_DIR / "audio"

def audio_to_mel_spectrogram(audio_path, sr=22050, n_mels=128, n_fft=2048,
                             hop_length=512, duration=None):
    """
    Convert audio file to Mel spectrogram (common CNN input format)

    Args:
        audio_path: Path to audio file
        sr: Sample rate (default 22050 Hz)
        n_mels: Number of Mel filter banks (default 128, corresponds to image height)
        n_fft: FFT window size
        hop_length: Hop length for STFT
        duration: Fixed duration in seconds, if specified will truncate or pad

    Returns:
        mel_spec: Mel spectrogram (n_mels, time_frames)
        y: Audio signal
        sr: Sample rate
    """
    # Load audio
    y, original_sr = librosa.load(str(audio_path), sr=sr)

    # Truncate or pad if fixed duration is specified
    if duration is not None:
        target_length = int(sr * duration)
        if len(y) > target_length:
            y = y[:target_length]
        elif len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')

    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmax=8000  # Typically focus on 0-8kHz
    )

    # Convert to log scale (Log-Mel Spectrogram, more commonly used)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db, y, sr


def process_training_sample(row_index=0, save_npy=True):
    """
    Process a training sample and convert to CNN input format

    Args:
        row_index: Data index
        save_npy: Whether to save as .npy file

    Returns:
        Dictionary containing mel_spectrogram, shape, class_id, class_name, file_name
    """
    print("=" * 60)
    print("Convert audio to CNN input format (Mel Spectrogram)")
    print("=" * 60)

    # Read training data
    df_train = pd.read_csv(TRAIN_CSV)
    row = df_train.iloc[row_index]

    file_name = row['slice_file_name']
    fold = row['fold']
    class_name = row['class']
    class_id = row['classID']

    print(f"\nProcessing sample:")
    print(f"  File name: {file_name}")
    print(f"  Class: {class_name} (ID: {class_id})")
    print(f"  Fold: {fold}")

    # Build audio path
    audio_path = AUDIO_DIR / f"fold{fold}" / file_name

    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}")
        return None

    # Convert to Mel spectrogram
    print(f"\nConverting to Mel spectrogram...")
    print(f"  Parameters: n_mels=128, sr=22050, hop_length=512")

    mel_spec, y, sr = audio_to_mel_spectrogram(
        audio_path,
        sr=22050,      # Standard sample rate
        n_mels=128,    # 128 Mel filter banks (image height)
        hop_length=512 # Hop length
    )

    print(f"\n✓ Conversion complete!")
    print(f"  Original audio length: {len(y) / sr:.2f} seconds")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Mel spectrogram shape: {mel_spec.shape}")
    print(f"    - Height (frequency dimension): {mel_spec.shape[0]} (Mel bins)")
    print(f"    - Width (time dimension): {mel_spec.shape[1]} (time frames)")

    # Normalize (usually needed for CNN training)
    mel_spec_norm = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
    print(f"\n  Normalized range: [{mel_spec_norm.min():.3f}, {mel_spec_norm.max():.3f}]")

    # Save as numpy array (can be directly used for training)
    if save_npy:
        npy_path = f"cnn_input_{row_index}.npy"
        np.save(npy_path, mel_spec_norm)
        print(f"✓ Saved as numpy array: {npy_path}")
        print(f"  File size: {Path(npy_path).stat().st_size / 1024:.2f} KB")
        print(f"  Can be loaded with np.load('{npy_path}') for CNN training")

    return {
        'mel_spectrogram': mel_spec_norm,
        'shape': mel_spec_norm.shape,
        'class_id': class_id,
        'class_name': class_name,
        'file_name': file_name
    }


def main():
    """Main function"""
    # Process first training sample
    result = process_training_sample(row_index=0, save_npy=True)

    if result:
        print("\n" + "=" * 60)
        print("Processing Summary")
        print("=" * 60)
        print(f"Output shape: {result['shape']}")
        print(f"Class: {result['class_name']} (ID: {result['class_id']})")
        print(f"File: {result['file_name']}")


if __name__ == "__main__":
    main()
# Finish your dataloader


# Finish your model class


# Train & Eval


# Submit your csv file to kaggle