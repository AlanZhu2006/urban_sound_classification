# 导入必要的模块 (如果之前没有导入)
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import gc

# 再次运行预测函数，但不带伪标签后缀
# 这将使用所有可用的 16 个模型 (8xCRNN + 8xB2) 进行集成预测，并生成 CSV。
print("\n📝 强制生成最终提交文件...")
generate_submission(prefix="")