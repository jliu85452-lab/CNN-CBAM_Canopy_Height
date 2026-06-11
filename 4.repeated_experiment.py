import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import json

from data_processing import GEDIWaveformDataset, train_test_split_dataset, create_dataloaders
from cnn_cbam_model import CNN_CBAM_CanopyHeight
from train import Trainer


class RepeatedExperiment:
    def __init__(self, n_experiments=3, seeds=[42, 123, 456]):
        """
        Args:
            n_experiments: 实验重复次数
        """
        self.n_experiments = n_experiments
        self.seeds = seeds[:n_experiments]
        self.results = {}
        
        # 创建保存目录
        os.makedirs('experiments', exist_ok=True)
        os.makedirs('figures', exist_ok=True)
        
    def run_experiment(self, exp_id, seed):
        """运行单次实验"""
        print(f"\n{'='*50}")
        print(f"实验 {exp_id+1}/{self.n_experiments} (随机种子: {seed})")
        print(f"{'='*50}")
        
        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 1. 加载数据
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        full_dataset = GEDIWaveformDataset(
            csv_file='gedi_data.csv',
            waveform_col='rx_waveform',
            target_col='CHM',
            waveform_length=1420
        )
        train_dataset, test_dataset = train_test_split_dataset(
            full_dataset, train_ratio=0.8, random_seed=seed
        )
        train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, batch_size=32)
        
        # 2. 创建模型
        model = CNN_CBAM_CanopyHeight(input_length=1420)
        
        # 3. 训练
        trainer = Trainer(model, device, save_dir=f'experiments/exp_{exp_id+1}')
        train_losses, val_losses, train_rmses, val_rmses = trainer.train(
            train_loader, test_loader, epochs=100, save_best=True
        )
        
        # 4. 加载最佳模型并测试
        checkpoint = torch.load(f'experiments/exp_{exp_id+1}/best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 测试集预测
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for waveforms, heights in test_loader:
                waveforms = waveforms.to(device)
                predictions = model(waveforms)
                all_preds.extend(predictions.cpu().numpy())
                all_targets.extend(heights.numpy())
        
        all_preds = np.array(all_preds).flatten()
        all_targets = np.array(all_targets).flatten()
        
        # 计算指标(可选用)
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(all_targets, all_preds)),
            'MAE': mean_absolute_error(all_targets, all_preds),
            'R2': r2_score(all_targets, all_preds),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_rmses': train_rmses,
            'val_rmses': val_rmses,
            'predictions': all_preds.tolist(),
            'targets': all_targets.tolist()
        }
        
        # 保存实验结果
        with open(f'experiments/exp_{exp_id+1}/metrics.json', 'w') as f:
            # 转换numpy为列表以便JSON序列化
            json_metrics = {k: v for k, v in metrics.items() 
                           if not isinstance(v, (np.ndarray, list)) or k in ['predictions', 'targets']}
            json_metrics['train_losses'] = train_losses
            json_metrics['val_losses'] = val_losses
            json_metrics['train_rmses'] = train_rmses
            json_metrics['val_rmses'] = val_rmses
            json.dump(json_metrics, f, indent=2)
        
        # 保存测试集预测CSV
        test_df = pd.DataFrame({
            'True_CHM': all_targets,
            'Predicted_CHM': all_preds,
            'Error': all_targets - all_preds
        })
        test_df.to_csv(f'experiments/exp_{exp_id+1}/你的test_predictions.csv', index=False)
        
        self.results[f'exp_{exp_id+1}'] = metrics
        
        return metrics
    
    def run_all_experiments(self):
        """运行所有重复实验"""
        for i, seed in enumerate(self.seeds):
            self.run_experiment(i, seed)
        
        # 汇总所有实验结果
        self.summarize_results()
        self.plot_all_loss_curves()
        self.plot_predictions_comparison()
        
    def summarize_results(self):
        """汇总三次实验的指标"""
        summary = {}
        for exp_name, metrics in self.results.items():
            summary[exp_name] = {
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R2': metrics['R2']
            }
        
        # 计算均值和标准差
        rmse_values = [m['RMSE'] for m in summary.values()]
        mae_values = [m['MAE'] for m in summary.values()]
        r2_values = [m['R2'] for m in summary.values()]
        
        summary['Mean ± Std'] = {
            'RMSE': f"{np.mean(rmse_values):.4f} ± {np.std(rmse_values):.4f}",
            'MAE': f"{np.mean(mae_values):.4f} ± {np.std(mae_values):.4f}",
            'R2': f"{np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}"
        }
        
        # 保存汇总表
        summary_df = pd.DataFrame(summary).T
        summary_df.to_csv('experiments/summary_results.csv')
        
        print("\n" + "="*50)
        print("三次实验结果汇总")
        print("="*50)
        print(summary_df)
        
        return summary_df
    
    def plot_all_loss_curves(self):
        """绘制所有实验的loss曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        colors = ['blue', 'red', 'green']
        
        for i, (exp_name, metrics) in enumerate(self.results.items()):
            epochs = range(1, len(metrics['train_losses']) + 1)
            
            # 训练损失
            axes[0].plot(epochs, metrics['train_losses'], 
                        color=colors[i], alpha=0.7, label=f'{exp_name} (Train)')
            axes[0].plot(epochs, metrics['val_losses'], 
                        color=colors[i], linestyle='--', alpha=0.7, label=f'{exp_name} (Val)')
            
            # 训练RMSE
            axes[1].plot(epochs, metrics['train_rmses'], 
                        color=colors[i], alpha=0.7, label=f'{exp_name} (Train)')
            axes[1].plot(epochs, metrics['val_rmses'], 
                        color=colors[i], linestyle='--', alpha=0.7, label=f'{exp_name} (Val)')
        
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss (MSE)', fontsize=12)
        axes[0].set_title('训练与验证损失曲线 (三次实验)', fontsize=14)
        axes[0].legend(loc='upper right', fontsize=8)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('RMSE (米)', fontsize=12)
        axes[1].set_title('训练与验证RMSE曲线 (三次实验)', fontsize=14)
        axes[1].legend(loc='upper right', fontsize=8)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figures/all_experiments_loss_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("损失曲线图已保存至: figures/all_experiments_loss_curves.png")
    
    def plot_predictions_comparison(self):
        """绘制三次实验的预测对比散点图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (exp_name, metrics) in enumerate(self.results.items()):
            targets = metrics['targets']
            predictions = metrics['predictions']
            
            # 散点图
            axes[i].scatter(targets, predictions, alpha=0.5, s=10)
            
            # 对角线
            min_val = min(min(targets), min(predictions))
            max_val = max(max(targets), max(predictions))
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect Prediction')
            
            # 添加指标文本
            rmse = metrics['RMSE']
            r2 = metrics['R2']
            axes[i].text(0.05, 0.95, f'RMSE: {rmse:.2f} m\nR²: {r2:.3f}', 
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            axes[i].set_xlabel('True CHM (m)', fontsize=10)
            axes[i].set_ylabel('Predicted CHM (m)', fontsize=10)
            axes[i].set_title(f'{exp_name}', fontsize=12)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc='lower right')
        
        plt.suptitle('三次实验预测值与真实值对比', fontsize=14)
        plt.tight_layout()
        plt.savefig('figures/predictions_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("预测对比图已保存至: figures/predictions_comparison.png")
    
    def plot_error_distribution(self):
        """绘制误差分布直方图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for i, (exp_name, metrics) in enumerate(self.results.items()):
            errors = np.array(metrics['targets']) - np.array(metrics['predictions'])
            
            axes[i].hist(errors, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
            axes[i].axvline(x=0, color='r', linestyle='--', label='Zero Error')
            axes[i].axvline(x=np.mean(errors), color='g', linestyle='-', label=f'Mean: {np.mean(errors):.2f}')
            
            axes[i].set_xlabel('Prediction Error (m)', fontsize=10)
            axes[i].set_ylabel('Frequency', fontsize=10)
            axes[i].set_title(f'{exp_name}\nStd: {np.std(errors):.2f} m', fontsize=12)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('三次实验预测误差分布', fontsize=14)
        plt.tight_layout()
        plt.savefig('figures/error_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("误差分布图已保存至: figures/error_distribution.png")


# 运行重复实验
if __name__ == "__main__":

    experiment = RepeatedExperiment(n_experiments=3, seeds=[42, 123, 456])
    
    experiment.run_all_experiments()
    
    experiment.plot_error_distribution()
    
    print("\n所有实验完成！结果保存在 'experiments/' 和 'figures/' 目录")