import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class GEDIWaveformDataset(Dataset):
      def __init__(self, csv_file, waveform_col='rx_waveform', target_col='CHM', 
                 waveform_length=1420, transform=None):
        self.data = pd.read_csv(csv_file)
        self.waveform_col = waveform_col
        self.target_col = target_col
        self.waveform_length = waveform_length
        self.transform = transform
        
        # 过滤无效数据（CHM或波形为空）
        self.data = self.data.dropna(subset=[waveform_col, target_col])
        
        print(f"加载数据: {len(self.data)} 个有效样本")
        print(f"CHM范围: {self.data[target_col].min():.2f} - {self.data[target_col].max():.2f} 米")
        
      def _process_waveform(self, waveform_str):
        """处理单个波形字符串为固定长度数组"""
        try:
            # 解析波形字符串（格式为 '一维数组'）
            if isinstance(waveform_str, str):
                waveform = np.array([float(x) for x in waveform_str.split(',')])
            else:
                waveform = np.array(waveform_str)
            
            current_len = len(waveform)
            if current_len < self.waveform_length:
                # 不足则末尾补0
                padded = np.zeros(self.waveform_length)
                padded[:current_len] = waveform
                return padded
            elif current_len > self.waveform_length:
                # 过长则截取中间1420个点（保留核心区域）
                start = (current_len - self.waveform_length) // 2
                return waveform[start:start + self.waveform_length]
            else:
                return waveform
        except Exception as e:
            # 异常情况返回全0
            print(f"波形处理错误: {e}")
            return np.zeros(self.waveform_length)
    
      def __len__(self):
        return len(self.data)
    
      def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 处理GEDI L1B波形
        waveform = self._process_waveform(row[self.waveform_col])
        
        # 归一化波形
        w_min, w_max = waveform.min(), waveform.max()
        if w_max - w_min > 1e-8:
            waveform = (waveform - w_min) / (w_max - w_min)
        else:
            waveform = np.zeros_like(waveform)
        
       
        chm_height = float(row[self.target_col])
        
        waveform = torch.FloatTensor(waveform).unsqueeze(0)  # [1, 1420]
        chm_height = torch.FloatTensor([chm_height])
        
        if self.transform:
            waveform = self.transform(waveform)
            
        return waveform, chm_height
def train_test_split_dataset(dataset, train_ratio=0.8, random_seed=42):
    np.random.seed(random_seed)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    split_point = int(train_ratio * dataset_size)
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    print(f"\n数据集划分完成（按正文80/20划分）:")
    print(f"  总样本数: {dataset_size}")
    print(f"  训练集: {len(train_dataset)} ({len(train_dataset)/dataset_size*100:.1f}%)")
    print(f"  测试集: {len(test_dataset)} ({len(test_dataset)/dataset_size*100:.1f}%)")
    
    return train_dataset, test_dataset


def create_dataloaders(train_dataset, test_dataset, batch_size=32):
    """创建DataLoader"""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                             shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# 使用示例
if __name__ == "__main__":
    dataset = GEDIWaveformDataset(
        csv_file='gedi_data.csv',      # 你的数据文件
        waveform_col='rx_waveform',     # 波形数据列
        target_col='CHM',               
        waveform_length=1420            
    )
    
    # 8:2 划分
    train_dataset, test_dataset = train_test_split_dataset(dataset, train_ratio=0.8)
    
    # 创建数据加载器
    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, batch_size=32)
    
    # 测试数据加载
    for batch_x, batch_y in train_loader:
        print(f"\n批次信息:")
        print(f"  波形形状: {batch_x.shape}")    
        print(f"  CHM标签: {batch_y.shape}")       
        print(f"  CHM值范围: {batch_y.min():.2f} - {batch_y.max():.2f} 米")
        break