代码文件说明

 1_data_processing.py
波形数据处理模块。读取 GEDI L1B 的一维序列数据（最好转成.CSV格式），将 rx_waveform 列的波形统一截取/填充为 1420 个点，并按 8:2 比例划分训练集和测试集。

 2_cnn_cbam_model.py
核心深度学习模型。包含多尺度并行卷积残差块、增强型 CBAM 注意力机制以及双注意力机制，以及完整的冠层高度回归网络。

 3_train.py
模型训练脚本：使用 Huber 损失函数、AdamW 优化器、余弦退火学习率调度，支持梯度裁剪、早停机制和 EMA 平均模型保存。

 4_repeated_experiments.py
重复实验与可视化：本次研究参考三次独立实验，以至于估测模型稳定性，保存每次的测试集预测结果和损失数据，并输出损失曲线、预测散点图、误差分布图。

airborne_tif_list
是3个研究区使用的机载数据

gedi_matched_pairs
3个研究区使用的GEDIL1B以及L2A数据

重合样本点说明：需要搜寻机载数据+L1B+L2A的重合点作为本次使用实验的样本点


# Python依赖库列表
# 安装命令: pip install -r requirements.txt

# GEDI 数据处理核心库
h5py>=3.0.0
numpy>=1.19.0
pandas>=1.2.0

# 地理空间处理
gdal>=3.0.0
geopandas>=0.9.0
shapely>=1.7.0

# 可视化
matplotlib>=3.3.0
tqdm>=4.62.0

# 核心深度学习框架
torch>=1.9.0
torchvision>=0.10.0

# 数据处理
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0

# 可视化
matplotlib>=3.3.0
seaborn>=0.11.0

# 进度条和日志
tqdm>=4.62.0
tensorboard>=2.7.0

# 其他工具
scipy>=1.6.0
joblib>=1.1.0

# 可选：GPU加速支持
# cudatoolkit>=11.3 
