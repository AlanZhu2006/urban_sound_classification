"""
数据集类模块
定义 PyTorch Dataset 类用于加载 Mel 频谱图数据
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, Union, List

from .config import CONFIG, NPY_DIR


class UrbanSoundDataset(Dataset):
    """
    城市声音数据集类
    
    支持:
    - RAM 缓存加速
    - 多分辨率输入
    - 训练/测试模式
    """
    
    def __init__(
        self, 
        df, 
        data_dir: Path = None, 
        mode: str = 'train', 
        target_mels: int = 128, 
        cache_data: bool = True
    ):
        """
        初始化数据集
        
        Args:
            df: 包含文件信息的 DataFrame
            data_dir: .npy 文件目录
            mode: 'train' 或 'test'
            target_mels: 目标 Mel 频率维度
            cache_data: 是否缓存数据到 RAM
        """
        self.df = df
        self.mode = mode
        self.target_mels = target_mels
        self.data_dir = data_dir if data_dir else NPY_DIR
        
        self.file_paths: List[Path] = []
        self.labels: List[int] = []
        self.filenames: List[str] = []
        
        # 构建文件路径列表
        for idx, row in df.iterrows():
            fname = str(row['slice_file_name'])
            npy_name = fname.replace('.wav', '.npy')
            if not npy_name.endswith('.npy'):
                npy_name += '.npy'
            
            self.filenames.append(fname)
            
            if mode == 'train':
                # 处理带 fold 信息的训练数据和伪标签数据
                if 'fold' in row and row['fold'] != -1:
                    folder = f"fold{row['fold']}"
                else:
                    folder = "test"
                
                p = self.data_dir / folder / npy_name
                # 容错: 尝试 test 目录
                if not p.exists():
                    p = self.data_dir / "test" / npy_name
                
                self.file_paths.append(p)
                self.labels.append(row['classID'])
            else:
                self.file_paths.append(self.data_dir / "test" / npy_name)
        
        # RAM 缓存
        self.cache_data = cache_data
        self.data_cache: List[Optional[np.ndarray]] = [None] * len(self.file_paths)
        
        if self.cache_data:
            self._preload_data()
    
    def _preload_data(self) -> None:
        """预加载所有数据到 RAM"""
        print(f"    ⚡ 预加载 {len(self.file_paths)} 个文件到 RAM...")
        for i, path in enumerate(tqdm(self.file_paths, desc="Loading RAM", leave=False)):
            try:
                self.data_cache[i] = np.load(path).astype(np.float32)
            except Exception:
                # 文件损坏时填充零矩阵
                self.data_cache[i] = np.zeros((128, 173), dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], 
                                              Tuple[torch.Tensor, int, str]]:
        """
        获取单个样本
        
        训练模式返回: (image_tensor, label)
        测试模式返回: (image_tensor, index, filename)
        """
        # 从缓存或磁盘加载
        if self.cache_data and self.data_cache[idx] is not None:
            mel = self.data_cache[idx]
        else:
            try:
                mel = np.load(self.file_paths[idx]).astype(np.float32)
            except Exception:
                mel = np.zeros((128, 173), dtype=np.float32)
        
        # 归一化到 [0, 1]
        delta = mel.max() - mel.min() + 1e-8
        img = (mel - mel.min()) / delta
        img_tensor = torch.tensor(img).unsqueeze(0)  # [1, H, W]
        
        # 调整分辨率
        if self.target_mels != img_tensor.shape[1]:
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),  # [1, 1, H, W]
                size=(self.target_mels, 173),
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)  # [1, H, W]
        
        if self.mode == 'train':
            return img_tensor, torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            return img_tensor, idx, self.filenames[idx]


def create_data_loaders(
    train_df, 
    val_df, 
    data_dir: Path = None,
    target_mels: int = 128,
    batch_size: int = None,
    num_workers: int = None,
    cache_data: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    
    Args:
        train_df: 训练集 DataFrame
        val_df: 验证集 DataFrame
        data_dir: 数据目录
        target_mels: Mel 频谱维度
        batch_size: 批次大小
        num_workers: 数据加载工作进程数
        cache_data: 是否缓存数据
    
    Returns:
        (train_loader, val_loader)
    """
    if batch_size is None:
        batch_size = CONFIG["BATCH_SIZE"]
    if num_workers is None:
        num_workers = CONFIG["NUM_WORKERS"]
    
    train_ds = UrbanSoundDataset(
        train_df, data_dir, 'train', target_mels, cache_data
    )
    val_ds = UrbanSoundDataset(
        val_df, data_dir, 'train', target_mels, cache_data
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size * 2, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_test_loader(
    test_df, 
    data_dir: Path = None,
    target_mels: int = 128,
    batch_size: int = None,
    num_workers: int = 2,
    cache_data: bool = True
) -> DataLoader:
    """
    创建测试数据加载器
    
    Args:
        test_df: 测试集 DataFrame
        data_dir: 数据目录
        target_mels: Mel 频谱维度
        batch_size: 批次大小
        num_workers: 工作进程数
        cache_data: 是否缓存
    
    Returns:
        test_loader
    """
    if batch_size is None:
        batch_size = CONFIG["BATCH_SIZE"] * 2
    
    test_ds = UrbanSoundDataset(
        test_df, data_dir, 'test', target_mels, cache_data
    )
    
    return DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
