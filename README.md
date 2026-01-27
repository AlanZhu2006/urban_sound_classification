# Urban Sound Classification | 城市声音分类

> **CSCI-SHU 360: Machine Learning Final Competition - Fall 2025**
> 
> **机器学习期末竞赛 - 2025秋季学期**
> 
> 🏆 **Kaggle Leaderboard: Rank 4 | Score: 0.88772**
> 
> 🏆 **Kaggle 排行榜: 第4名 | 分数: 0.88772**

A Multi-Resolution Ensemble Framework with Pseudo-Labeling based on CRNN for Urban Sound Classification.

基于 CRNN 的多分辨率集成框架，结合伪标签技术实现城市声音分类。

**Author / 作者:** Yanheng Zhu 朱彦恒 (yz11502@nyu.edu)

---

## 📋 Abstract | 摘要

This project presents a **Multi-Resolution Heterogeneous Ensemble Framework** with semi-supervised pseudo-labeling for urban sound classification. Addressing the challenges of limited labeled data and high inter-class acoustic similarity, we developed a three-stage pipeline:

本项目提出了一种**多分辨率异构集成框架**，结合半监督伪标签技术用于城市声音分类。针对标注数据有限和类间声学相似度高的挑战，我们开发了三阶段流水线：

1. **High-efficiency parallel preprocessing | 高效并行预处理** - Audio to Log-Mel Spectrogram conversion | 音频转 Log-Mel 频谱图
2. **Heterogeneous teacher training with pseudo-labeling | 异构教师训练 + 伪标签生成** - CRNN + EfficientNet ensemble | CRNN + EfficientNet 集成
3. **Dense multi-resolution ensemble inference | 密集多分辨率集成推理** - With dynamic conflict resolution | 动态冲突解决

---

## 🔄 Pipeline Overview | 流水线概述

```
┌──────────────────────────────────────────────────────────────────┐
│  Stage 1: preprecess.py                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ • Audio resampling to 22,050 Hz | 音频重采样至 22,050 Hz    │  │
│  │ • Duration normalization to 4.0s | 时长标准化为 4.0 秒      │  │
│  │ • Log-Mel Spectrogram (128×173) | Log-Mel 频谱图提取       │  │
│  │ • Parallel processing (joblib) | 并行处理                  │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  Stage 2: train.py (Teacher Phase)                               │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ • Heterogeneous Ensemble | 异构集成: CRNN + EffNet-B2      │  │
│  │ • Multi-resolution | 多分辨率: 64, 96, 128 Mel bins        │  │
│  │ • 8-Fold Cross Validation | 8折交叉验证                    │  │
│  │ • Hill Climbing optimization | 爬山算法权重优化            │  │
│  │ • Pseudo-label (Conf > 0.90) | 伪标签 (置信度 > 0.90)      │  │
│  │ → Output | 输出: train_pseudo_v3.csv                       │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  Stage 3: main.py (Student Phase - Final Submission)             │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ • Dense multi-res | 密集多分辨率: 128,160,192,256,320      │  │
│  │ • Train with pseudo-labels | 使用伪标签数据训练            │  │
│  │ • Test-Time Augmentation (TTA) | 测试时增强                │  │
│  │ • Post-process | 后处理: Power Sharpening + 类别权重       │  │
│  │ • Dynamic Conflict Resolution | 动态冲突解决               │  │
│  │ → Output | 输出: submission.csv                            │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Preprocessing | 数据预处理

### Audio Standardization | 音频标准化
| Parameter 参数 | Value 值 | Rationale 原理 |
|----------------|----------|----------------|
| Sample Rate 采样率 | 22,050 Hz | Preserves freq up to ~11 kHz (Nyquist) 保留至 11kHz 频率 |
| Duration 时长 | 4.0 seconds 秒 | Fixed shape; pad/truncate 固定张量形状，短补长截 |
| Max Frequency 最高频率 | 8,000 Hz | Perceptually relevant bands 感知相关频段 |

### Log-Mel Spectrogram Extraction | Log-Mel 频谱图提取
| Parameter 参数 | Value 值 | Description 描述 |
|----------------|----------|------------------|
| n_fft | 2048 | Large window for high freq resolution 大窗口高频率分辨率 |
| hop_length | 512 | Determines temporal resolution 决定时间分辨率 |
| n_mels | 128 | Number of Mel bands Mel 频带数量 |
| Output Shape 输出形状 | (128, 173) | Height × Width 高度 × 宽度 |

Features are serialized as `.npy` (float32) files using parallel processing (`joblib`, `n_jobs=-1`).

特征序列化为 `.npy` (float32) 文件，使用并行处理 (`joblib`, `n_jobs=-1`)。

---

## 🏗️ Model Architectures | 模型架构

### 1. Advanced CRNN (Convolutional Recurrent Neural Network) | 高级卷积循环神经网络

Designed to capture **temporal dynamics** with multi-stage regularization.

专为捕获**时序动态特征**设计，采用多阶段正则化。

```
Input [1, H, W]
    ↓
┌─────────────────────────────────────┐
│ Stem: Conv2d(1→64) + BN + ReLU      │
│       + MaxPool2d(2)                │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ SE-ResBlock(64→64) + MaxPool2d(2)   │  ← Squeeze-and-Excitation
├─────────────────────────────────────┤     Channel Attention
│ SE-ResBlock(64→128) + MaxPool2d(2)  │
├─────────────────────────────────────┤
│ SE-ResBlock(128→256) + MaxPool2d    │
├─────────────────────────────────────┤
│ SE-ResBlock(256→512) + MaxPool2d    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Frequency Mean Pooling → [B, 512, T]│
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Bi-Directional GRU (2 layers)       │
│ 512 → 256×2, dropout=0.2            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Linear Attention Mechanism          │
│ αₜ = softmax(Wᵀhₜ + b)              │
│ c = Σ αₜhₜ                          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Dropout(0.3) + FC(512→10)           │
└─────────────────────────────────────┘
```

**Key Components | 核心组件:**
- **SE-ResBlocks (SE残差块)**: Adaptively recalibrate channel-wise responses | 自适应重校准通道响应
- **Bi-GRU with Dropout (双向GRU)**: Model sequential audio, prevent overfitting | 建模音频时序，防止过拟合
- **Linear Attention (线性注意力)**: Focus on active segments, ignore silence | 聚焦活跃片段，忽略静音

### 2. LightweightEffNet (EfficientNet-B2 Style) | 轻量 EfficientNet

Focuses on **spatial spectral textures** with GeM pooling.

专注于**空间频谱纹理**特征，采用 GeM 池化。

**GeM (Generalized Mean) Pooling | 广义均值池化:**

$$f(x) = \left( \frac{1}{|X|} \sum_{x \in X} x^p \right)^{1/p}$$

- As $p \to \infty$, GeM approaches Max Pooling | 当 $p \to \infty$ 时，趋近最大池化
- Learnable parameter adapts pooling strategy | 可学习参数自适应池化策略
- Superior for short events & background | 对短事件（枪声）和背景纹理均有效

---

## 🔬 Semi-Supervised Self-Training | 半监督自训练 (伪标签)

To overcome data scarcity, we implement an iterative self-training loop:

为克服数据稀缺问题，我们实现了迭代自训练循环：

### Step 1: Teacher Training & Hill Climbing | 第一步：教师训练与爬山优化
- Train initial ensemble on labeled data (8-fold CV) | 在标注数据上训练初始集成模型（8折交叉验证）
- Use Hill Climbing to optimize ensemble weights on OOF predictions | 使用爬山算法优化 OOF 预测的集成权重

### Step 2: High-Confidence Pseudo-Labeling | 第二步：高置信度伪标签
```python
D_pseudo = {(xᵢ, ŷᵢ) | max(P(y|xᵢ)) > 0.90}
```
Only samples with confidence > 0.90 are accepted as pseudo-labels.

仅接受置信度 > 0.90 的样本作为伪标签。

### Step 3: Dataset Expansion & Retraining | 第三步：数据集扩展与重训练
- Merge pseudo-labeled samples with original training data | 将伪标签样本与原始训练数据合并
- Retrain models on expanded distribution (train_pseudo_v3.csv) | 在扩展分布上重训练模型

---

## 🎨 Data Augmentation | 数据增强

### Mixup and CutMix Regularization | Mixup 与 CutMix 正则化

**Mixup** (λ ~ Beta(α, α)):
$$\tilde{x} = \lambda x_i + (1-\lambda) x_j$$
$$\tilde{y} = \lambda y_i + (1-\lambda) y_j$$

**CutMix** - Replace rectangular region | 替换矩形区域:
$$r_w = W\sqrt{1-\lambda}, \quad r_h = H\sqrt{1-\lambda}$$

### SpecAugment | 频谱增强
- **Time Masking 时间遮蔽**: Randomly mask vertical strips | 随机遮蔽垂直条带
- **Frequency Masking 频率遮蔽**: Randomly mask horizontal strips | 随机遮蔽水平条带

### Gaussian Noise Injection | 高斯噪声注入
- Add $\mathcal{N}(0, \sigma^2)$ noise with probability 0.5 | 以 0.5 概率添加高斯噪声
- Simulates sensor noise, improves robustness | 模拟传感器噪声，提升鲁棒性

---

## 📈 Training Objective: Focal Loss | 训练目标：Focal Loss

To address hard-to-classify samples and class imbalance:

为解决难分类样本和类别不平衡问题：

$$FL(p_t) = -\alpha(1-p_t)^\gamma \log(p_t)$$

| Parameter 参数 | Value 值 | Effect 效果 |
|----------------|----------|-------------|
| γ (gamma) | 2.0 | Down-weight easy examples 降低简单样本权重 |
| α (alpha) | 1.0 | Class balance factor 类别平衡因子 |
| label_smoothing 标签平滑 | 0.05 | Prevent overconfidence 防止过度自信 |

---

## 📊 Ablation Study: Spectral Resolution | 消融实验：频谱分辨率

| Resolution 分辨率 | Avg. Val Accuracy 平均验证准确率 | Ensemble Weight 集成权重 | Rationale 原理 |
|-------------------|----------------------------------|--------------------------|----------------|
| 128 | 81.86% | 0.6 | Baseline regularizer 基线正则化；缺乏细节 |
| 160 | 81.72% | 0.9 | Intermediate scale 中间尺度 |
| 192 | 83.10% | 1.3 | Significant info gain 显著信息增益 |
| **256** | **83.78%** | **1.5** | **Peak Performance 最佳性能** |
| 320 | 81.98% | 1.2 | Diminishing returns 收益递减 |

---

## 🔧 Post-Processing Pipeline | 后处理流水线

### 1. Power Sharpening | 幂次锐化
```python
probs = probs ** 1.05  # Enhance high-confidence predictions 增强高置信度预测
probs = probs / probs.sum(dim=1, keepdim=True)  # Re-normalize 重新归一化
```

### 2. Static Class Weighting | 静态类别加权

Based on Confusion Matrix analysis to balance Precision/Recall:

基于混淆矩阵分析，平衡精确率/召回率：

```python
# [AirCon 空调, CarHorn 汽车喇叭, Children 儿童, Dog 狗叫, Drilling 钻孔, 
#  Engine 引擎, GunShot 枪声, Jackhammer 破碎锤, Siren 警笛, Music 音乐]
W_class = [1.85, 1.00, 0.85, 0.95, 1.25, 1.10, 1.00, 0.65, 1.35, 0.55]
```

| Class 类别 | Weight 权重 | Rationale 原理 |
|------------|-------------|----------------|
| Air Conditioner 空调 (0) | 1.85 | ↑ Boost low recall 提升低召回率 (高假阴性) |
| Jackhammer 破碎锤 (7) | 0.65 | ↓ Suppress over-predict 抑制过度预测 |
| Street Music 音乐 (9) | 0.55 | ↓ Suppress "broad attractor" 抑制"广泛吸引器" (高假阳性) |

### 3. Dynamic Conflict Resolution | 动态冲突解决

Calibrate decision boundaries for systematic biases:

校准决策边界以应对系统性偏差：

| Confusion Pair 混淆对 | Condition 条件 | Correction 修正 |
|------------------------|----------------|-----------------|
| Engine(5) → AirCon(0) | `probs[0] > probs[5] * 0.60` | `probs[0] *= 1.5` |
| Music(9) → Children(2) | `probs[2] > probs[9] * 0.70` | `probs[2] *= 1.5` |
| Music(9) → Dog(3) | `probs[3] > probs[9] * 0.70` | `probs[3] *= 1.5` |
| Jackhammer(7) → Drilling(4) | `probs[4] > probs[7] * 0.75` | `probs[4] *= 1.4` |

---

## 📁 Project Structure | 项目结构

```
├── main.py                 # Stage 3: Final inference 最终推理与提交
├── train.py                # Stage 2: Ensemble training 异构集成训练 + 伪标签
├── model.py                # Lightweight inference 轻量推理版本 (备用)
├── preprecess.py           # Stage 1: Audio preprocessing 音频预处理
├── run.py                  # Local entry point 本地运行入口
├── README.md               # Project documentation 项目说明
│
├── src/                    # Modular source 模块化源代码
│   ├── __init__.py         # Package init 包初始化与导出
│   ├── config.py           # Config params 配置参数 (训练/推理分离)
│   ├── preprocessing.py    # Audio preprocess 音频预处理
│   ├── dataset.py          # Dataset class 数据集类 (RAM缓存)
│   ├── augmentation.py     # Data augmentation 数据增强
│   ├── models.py           # Model definitions 模型定义
│   ├── train.py            # Training functions 训练函数
│   ├── inference.py        # Inference + postprocess 推理 + 后处理
│   ├── evaluate.py         # Evaluation 评估函数
│   └── utils.py            # Utilities 工具函数
│
├── Kaggle_Data/            # Local data 本地数据目录
│   ├── audio/              # Raw audio 原始音频 (fold1-8, test)
│   ├── metadata/           # CSV metadata CSV 元数据
│   └── processed_npy/      # Preprocessed Mel 预处理的 Mel 频谱
│
├── models/                 # Saved weights 本地保存的模型权重
└── CRNN_Auto_Weakness/     # Pretrained weights 预训练模型权重
```

---

## 🎛️ Configuration Comparison | 配置对比

| Config 配置项 | train.py (Teacher) | main.py (Student) |
|---------------|-------------------|-------------------|
| Resolution 分辨率 | [64, 96, 128] | [128, 160, 192, 256, 320] |
| Model 模型 | CRNN + EffNet | CRNN only |
| Train CSV 训练CSV | kaggle_train.csv | train_pseudo_r3.csv |
| Epochs 训练轮数 | 50 | 55 |
| Pseudo-label 伪标签 | Generate 生成 | Use 使用 |

---

## 🚀 Quick Start | 快速开始

### Kaggle Environment | Kaggle 环境
```python
# 1. Preprocessing (run once) | 预处理 (运行一次)
!python preprecess.py

# 2. Training + pseudo-label generation | 训练 + 伪标签生成
!python train.py

# 3. Final inference | 最终推理
!python main.py
```

### Local Environment | 本地环境
```bash
# Use modular entry point | 使用模块化入口
python run.py --mode all
```

---

## 📦 Kaggle Directory Structure | Kaggle 目录结构

```
/kaggle/input/
├── kaggle-data/          # Raw data 原始数据
├── processed/            # Preprocessed .npy 预处理的.npy文件
├── new-csv/              # Pseudo-label CSV 伪标签CSV (train_pseudo_r3.csv)
├── pth-final/            # Pretrained weights 预训练权重
└── final-pseudo/         # Pseudo-trained weights 伪标签训练的权重

/kaggle/working/
└── models/               # Runtime copied weights 运行时复制的权重
```

---

## 📚 References | 参考文献

- UrbanSound8K Dataset (Salamon et al., 2014)
- EfficientNet (Tan & Le, 2019)
- Focal Loss (Lin et al., 2017)
- SpecAugment (Park et al., 2019)
- Mixup (Zhang et al., 2018)

---

## License | 许可证

MIT License
