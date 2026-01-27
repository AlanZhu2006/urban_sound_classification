"""
模型定义模块
包含 CRNN, LightweightEffNet 等模型架构
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import NUM_CLASSES


# ==========================================
# 基础组件
# ==========================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    通道注意力机制
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResBlock(nn.Module):
    """
    残差块 + SE 注意力
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.se = SEBlock(out_channels)
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return F.relu(out)


class GeM(nn.Module):
    """
    Generalized Mean Pooling
    比 AvgPool 和 MaxPool 更灵活的池化方式
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


# ==========================================
# 损失函数
# ==========================================

class FocalLoss(nn.Module):
    """
    Focal Loss
    解决类别不平衡问题
    """
    def __init__(
        self, 
        alpha: float = 1.0, 
        gamma: float = 2.0, 
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs, targets, 
            reduction='none', 
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ==========================================
# 模型 A: Advanced CRNN
# ==========================================

class AdvancedCRNN(nn.Module):
    """
    高级 CRNN 模型
    
    架构:
    - CNN 主干: ResBlock + SE 注意力
    - RNN: 双向 GRU
    - 注意力池化
    """
    def __init__(
        self, 
        num_classes: int = NUM_CLASSES, 
        input_mels: int = 128,
        hidden_size: int = 256,
        num_gru_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.input_mels = input_mels
        
        # CNN 特征提取
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.layer1 = nn.Sequential(ResBlock(64, 64), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(ResBlock(64, 128), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(ResBlock(128, 256), nn.MaxPool2d((2, 1)))
        self.layer4 = nn.Sequential(ResBlock(256, 512), nn.MaxPool2d((2, 1)))
        
        # RNN
        self.gru = nn.GRU(
            512, hidden_size, 
            batch_first=True, 
            bidirectional=True,
            num_layers=num_gru_layers,
            dropout=dropout if num_gru_layers > 1 else 0
        )
        
        # 注意力层
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # 分类头
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN
        x = self.stem(x)      # [B, 64, H/2, W/2]
        x = self.layer1(x)    # [B, 64, H/4, W/4]
        x = self.layer2(x)    # [B, 128, H/8, W/8]
        x = self.layer3(x)    # [B, 256, H/16, W/8]
        x = self.layer4(x)    # [B, 512, H/32, W/8]
        
        # 频率维度平均, 准备 RNN 输入
        x = x.mean(dim=2)              # [B, 512, T]
        x = x.permute(0, 2, 1)         # [B, T, 512]
        
        # RNN
        x, _ = self.gru(x)             # [B, T, hidden*2]
        
        # 注意力池化
        weights = F.softmax(self.attention(x), dim=1)  # [B, T, 1]
        x = (x * weights).sum(dim=1)                    # [B, hidden*2]
        
        # 分类
        x = self.dropout(x)
        return self.fc(x)


# ==========================================
# 模型 B: Lightweight EfficientNet-style
# ==========================================

class LightweightEffNet(nn.Module):
    """
    轻量级 EfficientNet 风格模型
    
    使用深度可分离卷积减少参数量
    """
    def __init__(
        self, 
        num_classes: int = NUM_CLASSES, 
        input_mels: int = 128,
        dropout: float = 0.4
    ):
        super().__init__()
        
        self.input_mels = input_mels
        
        self.features = nn.Sequential(
            # Stem
            nn.Conv2d(1, 32, 3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            
            # Blocks
            self._make_block(32, 64, stride=2),
            self._make_block(64, 128, stride=2),
            self._make_block(128, 256, stride=2),
            self._make_block(256, 512, stride=2),
        )
        
        self.gem = GeM()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def _make_block(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1
    ) -> nn.Sequential:
        """
        创建 MBConv 风格的块
        深度可分离卷积 + SE 注意力
        """
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, 3, 
                      padding=1, stride=stride, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            
            # SE
            SEBlock(in_channels),
            
            # Pointwise
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gem(x).flatten(1)
        return self.classifier(x)


# ==========================================
# 模型工厂
# ==========================================

def create_model(
    model_type: str = "CRNN",
    num_classes: int = NUM_CLASSES,
    input_mels: int = 128,
    pretrained_path: Optional[str] = None
) -> nn.Module:
    """
    创建模型
    
    Args:
        model_type: "CRNN" 或 "EffNet"
        num_classes: 分类数
        input_mels: Mel 频谱维度
        pretrained_path: 预训练权重路径
    
    Returns:
        模型实例
    """
    if model_type.upper() == "CRNN":
        model = AdvancedCRNN(num_classes=num_classes, input_mels=input_mels)
    elif model_type.upper() in ["EFFNET", "EFFICIENTNET"]:
        model = LightweightEffNet(num_classes=num_classes, input_mels=input_mels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location='cpu')
        # 处理 torch.compile 产生的键名前缀
        new_state_dict = {
            k.replace('_orig_mod.', ''): v 
            for k, v in state_dict.items()
        }
        model.load_state_dict(new_state_dict)
    
    return model


def get_model_count(model: nn.Module) -> int:
    """获取模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
