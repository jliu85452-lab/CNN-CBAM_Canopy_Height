import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelEMA:
    """
    模型指数移动平均（EMA）
    用于保存模型，提升泛化能力
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # 初始化影子参数
        for name, param in model.state_dict().items():
            self.shadow[name] = param.clone().detach()
    
    def update(self):
        """更新平均模型参数"""
        for name, param in self.model.state_dict().items():
            new_average = (1.0 - self.decay) * param.detach() + self.decay * self.shadow[name]
            self.shadow[name] = new_average.clone().detach()
    
    def apply_shadow(self):
        """应用平均参数到模型"""
        for name, param in self.model.state_dict().items():
            self.backup[name] = param.clone().detach()
            param.data.copy_(self.shadow[name])
    
    def restore(self):
        """恢复原始模型参数"""
        for name, param in self.model.state_dict().items():
            param.data.copy_(self.backup[name])
        self.backup = {}
    
    def save_average_model(self, save_path):
        """保存模型"""
        self.apply_shadow()
        torch.save(self.model.state_dict(), save_path)
        self.restore()
        print(f"平均模型已保存至: {save_path}")


def train_with_average_model(model, train_loader, val_loader,
                              y_scaler, fold_save_path,
                              epochs=500,patience=50, ema_decay=0.999):  
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # 初始化EMA
    ema = ModelEMA(model, decay=ema_decay)
    
    best_r = -float('inf')
    best_ema_r = -float('inf')
    best_model_state = None
    best_ema_state = None
    counter = 0

    print(f"  开始训练 (最大{epochs}轮, 早停patience={patience}, EMA衰减={ema_decay})")

    for epoch in range(epochs):
        # ========== 训练阶段 ==========
        model.train()
        train_loss, train_batches = 0, 0
        
        for waveforms, targets in train_loader:
            waveforms, targets = waveforms.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(waveforms)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 更新EMA
            ema.update()
            
            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0

        # ========== 验证阶段（使用原始模型） ==========
        model.eval()
        val_preds_norm, val_targets_norm = [], []
        
        with torch.no_grad():
            for waveforms, targets in val_loader:
                waveforms = waveforms.to(device)
                outputs = model(waveforms)
                
                pred = outputs.cpu().numpy()
                if pred.ndim == 0:
                    pred = np.array([pred])
                val_preds_norm.extend(pred)
                val_targets_norm.extend(targets.numpy())

        if val_preds_norm:
            # 反标准化到真实尺度
            val_preds = y_scaler.inverse_transform(np.array(val_preds_norm).reshape(-1, 1)).flatten()
            val_targets = y_scaler.inverse_transform(np.array(val_targets_norm).reshape(-1, 1)).flatten()
            
            # 计算相关系数 R
            if len(val_preds) > 1 and np.std(val_preds) > 0 and np.std(val_targets) > 0:
                current_r = np.corrcoef(val_preds, val_targets)[0, 1]
            else:
                current_r = 0
            
            current_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        else:
            current_r, current_rmse = 0, 0

        # ========== 验证阶段（使用EMA平均模型） ==========
        ema.apply_shadow()
        ema_preds_norm, ema_targets_norm = [], []
        
        with torch.no_grad():
            for waveforms, targets in val_loader:
                waveforms = waveforms.to(device)
                outputs = model(waveforms)
                
                pred = outputs.cpu().numpy()
                if pred.ndim == 0:
                    pred = np.array([pred])
                ema_preds_norm.extend(pred)
                ema_targets_norm.extend(targets.numpy())
        
        ema.restore()
        
        if ema_preds_norm:
            ema_preds = y_scaler.inverse_transform(np.array(ema_preds_norm).reshape(-1, 1)).flatten()
            ema_targets = y_scaler.inverse_transform(np.array(ema_targets_norm).reshape(-1, 1)).flatten()
            
            if len(ema_preds) > 1 and np.std(ema_preds) > 0 and np.std(ema_targets) > 0:
                ema_r = np.corrcoef(ema_preds, ema_targets)[0, 1]
            else:
                ema_r = 0
            
            ema_rmse = np.sqrt(mean_squared_error(ema_targets, ema_preds))
        else:
            ema_r, ema_rmse = 0, 0

        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}: Loss={avg_train_loss:.4f}, "
                  f"Model R={current_r:.4f}/{current_rmse:.2f}m, "
                  f"EMA R={ema_r:.4f}/{ema_rmse:.2f}m")

        # 早停检查
        if current_r > best_r + 0.001:
            best_r = current_r
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
        
       
        if ema_r > best_ema_r:
            best_ema_r = ema_r
            best_ema_state = {k: v.clone() for k, v in ema.shadow.items()}

        # 更新学习率
        scheduler.step()

        if counter >= patience:
            print(f"    早停触发！最佳Model R={best_r:.4f}, 最佳EMA R={best_ema_r:.4f}")
            break

    # 保存平均模型（EMA）
    if best_ema_state is not None:
        # 恢复最佳EMA状态并保存
        for name, param in model.state_dict().items():
            param.data.copy_(best_ema_state[name])
        torch.save(model.state_dict(), fold_save_path)
        print(f"    模型已保存 (EMA R={best_ema_r:.4f})")
    else:
               ema.apply_shadow()
    torch.save(model.state_dict(), fold_save_path)
    ema.restore()
    print(f"    当前模型已保存")

    return model, ema, best_ema_r


def run_single_training(model, train_loader, val_loader, y_scaler, 
                        save_path='checkpoints/average_model.pth'):
    """单次训练入口"""
    print("\n=== 开始单次训练（使用EMA平均模型） ===")
    trained_model, ema, best_r = train_with_average_model(
        model, train_loader, val_loader, y_scaler, save_path,
        epochs=100, patience=20, ema_decay=0.999
    )
    print(f"训练完成！最佳平均模型R: {best_r:.4f}")
    return trained_model, ema, best_r


# ========== 主程序 ==========
if __name__ == "__main__":
    from data_processing import GEDIWaveformDataset, train_test_split_dataset, create_dataloaders
    from cnn_cbam_model import CNN_CBAM_CanopyHeight
    
    os.makedirs('checkpoints', exist_ok=True)
    
    # 加载数据
    dataset = GEDIWaveformDataset(
        csv_file='gedi_data.csv',
        waveform_col='rx_waveform',
        target_col='CHM',
        waveform_length=1420
    )
    
    train_dataset, test_dataset = train_test_split_dataset(dataset, train_ratio=0.8)
    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, batch_size=32)
    
    # 准备y_scaler
    train_chm = [dataset[i][1].item() for i in train_dataset.indices]
    y_scaler = StandardScaler()
    y_scaler.fit(np.array(train_chm).reshape(-1, 1))
    
    # 创建模型
    model = CNN_CBAM_CanopyHeight(input_channels=1, input_length=1420)
    model = model.to(device)
    
    # 训练并保存模型
    trained_model, ema, best_r = run_single_training(
        model, train_loader, test_loader, y_scaler,
        save_path='checkpoints/average_model.pth'
    )
    
    print(f"\n✅ 训练完成！模型已保存至 checkpoints/average_model.pth")