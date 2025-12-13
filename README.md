# 联邦学习电池分类系统 (Federated Learning Battery Classification System)

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Federated Learning](https://img.shields.io/badge/Federated%20Learning-PyTorch-orange.svg)](https://pytorch.org)

## 📖 项目简介

这是一个基于联邦学习的电池分类系统，旨在通过分布式机器学习方法对电池进行分类和识别。系统支持多种电池类型的分类，包括不同容量和化学成分的电池，保护数据隐私的同时实现高效的模型训练。

## ✨ 功能特点

- 🔄 **联邦学习架构**: 支持多客户端分布式训练，保护数据隐私
- 🔋 **多电池类型支持**: 支持8种不同类型的电池分类
  - 10Ah_LMO, 15Ah_NMC, 21Ah_NMC, 24Ah_LMO
  - 25Ah_LMO, 26Ah_LMO, 35Ah_LFP, 68Ah_LFP
- 🤖 **多种机器学习模型**: MLP、随机森林、KNN、高斯过程、决策树
- 📊 **数据增强技术**: 支持多种数据增强策略
- 🎯 **智能聚合算法**:
  - Avg (平均聚合)
  - 投票机制聚合
  - 马氏距离加权聚合
  - 编码器距离加权聚合
- 📈 **可视化分析**: 提供详细的性能分析和可视化结果
- 🔬 **实验框架**: 支持100次随机实验重复，自动生成报告

## 项目结构

```
github/
├── central.py              # 中央化模型训练脚本
├── client_model.py         # 客户端模型类定义
├── dataset.py              # 数据集处理和客户端生成
├── run.py                  # 联邦学习主运行脚本
├── reat_dataset.py         # 数据集创建脚本
├── split_dataset.py        # 数据集分割工具
├── plot.py                 # 结果可视化工具
├── exp/                    # 实验模块
│   ├── encoder.py          # 自编码器实现
│   └── evaluation.py       # 模型评估工具
└── distance/               # 距离计算和聚合方法
    ├── aggregate_fed.py    # FedAvg聚合
    ├── aggregate_1.py      # 投票聚合
    ├── Ma_distance1.py     # 马氏距离加权
    └── novel_distance.py   # 编码器距离加权
```

## 🛠️ 安装要求

确保您的环境中已安装以下Python包：

```bash
# 基础依赖
pip install pandas>=1.3.0
pip install numpy>=1.20.0
pip install scikit-learn>=1.0.0
pip install torch>=1.9.0
pip install matplotlib>=3.3.0
pip install seaborn>=0.11.0
pip install scipy>=1.7.0

# 或使用requirements.txt一键安装
pip install -r requirements.txt
```

### 系统要求

- **Python**: 3.7+
- **操作系统**: Windows, macOS, Linux
- **内存**: 建议4GB+
- **存储**: 至少2GB可用空间

## 🚀 快速开始

### 方法一：使用快速启动脚本

```bash
# 运行交互式快速开始
python example_usage.py
```

### 方法二：手动执行完整流程

#### 1. 克隆项目

```bash
git clone https://github.com/yourusername/federated-battery-classification.git
cd federated-battery-classification
```

#### 2. 安装依赖

```bash
pip install -r requirements.txt
```

#### 3. 数据准备

首先运行数据集创建脚本来生成训练和测试数据：

```bash
python reat_dataset.py
```

#### 4. 运行完整联邦学习实验

```bash
# 运行联邦学习主程序（自动完成所有步骤）
python run.py
```

#### 5. 中央化模型训练（对比实验）

```bash
python central.py
```

### 📊 实验配置

可以通过修改 `run.py` 顶部的配置参数来调整实验：

```python
MODEL_NAME = "MLP"          # 模型类型: "MLP", "RandomForest", "KNN", "GaussianProcess", "DecisionTree"
NUM_CLIENT = 6              # 客户端数量
NUM_EXPERIMENTS = 100       # 实验次数
MINI_TYPE = 2               # 每个客户端最少电池类型数
MAX_TYPE = 5                # 每个客户端最多电池类型数
HIDDEN_DIM = 10             # 编码器隐藏层维度
```

## 核心算法

### 客户端模型 (client_model.py)

- **数据增强**: 提供两种数据增强策略
  - 简单复制增强到固定数量
  - 基于协方差矩阵的噪声增强

- **模型训练**: 支持多种机器学习算法
  - MLP (多层感知机)
  - RandomForest (随机森林)
  - KNN (K近邻)
  - GaussianProcess (高斯过程)
  - DecisionTree (决策树)

- **编码器训练**: 为每个电池类型训练独立的自编码器

### 聝合策略

1. **Avg (平均聚合)**: 概率平均的聚合算法
2. **投票机制**: 基于客户端预测的投票聚合
3. **马氏距离加权**: 基于距离的智能权重分配
4. **编码器距离加权**: 利用自编码器的OOD检测能力

## 实验结果

系统会自动生成以下分析结果：

- 各客户端的分类准确率
- 混淆矩阵可视化
- 不同聚合方法的性能对比
- 每种电池类型的详细分类报告

## 参数配置

主要参数说明：

- `num_client`: 客户端数量 (默认: 6)
- `mini_type`: 每个客户端最少电池类型数 (默认: 2)
- `max_type`: 每个客户端最多电池类型数 (默认: 5)
- `model_type`: 使用的机器学习模型 (默认: "MLP")
- `hidden_dim`: 编码器隐藏层维度 (默认: 10)

## 📁 输出文件结构

运行实验后，系统会生成以下文件和目录：

```
github/
├── 📁 data/                    # 数据文件
│   ├── test0.csv              # 测试数据
│   ├── test1.csv
│   └── ...
├── 📁 client_model/            # 客户端模型和结果
│   ├── 0/                     # 实验0的结果
│   │   ├── client_0.pkl       # 客户端0模型
│   │   ├── client_1.pkl       # 客户端1模型
│   │   └── ...
│   ├── 1/                     # 实验1的结果
│   └── ...
├── 📁 results/                 # 实验结果汇总
│   └── federated_learning_results.csv
├── 📁 plots/                   # 可视化图表
└── 📄 *.png                   # 生成的图表文件
```

## 🎯 实验结果解读

### 性能指标

系统会自动计算以下性能指标：

- **准确率 (Accuracy)**: 整体分类准确率
- **F1分数**: 各电池类型的F1分数
- **混淆矩阵**: 详细的分类错误分析
- **客户端相似度**: 客户端间数据分布相似性

### 聚合方法对比

| 方法 | 描述 | 适用场景 |
|------|------|----------|
| Avg | 平均聚合 (Average Aggregation) | 基础概率平均聚合方法 |
| Voting | 投票机制聚合 | 多模型集成决策 |
| Ma_Distance | 马氏距离加权 | 数据分布不均衡场景 |
| Encoder_Softmax | 编码器加权(softmax) | 高维特征空间 |
| Encoder_Argmin | 编码器加权(argmin) | 异常检测场景 |

## 🐛 故障排除

### 常见问题

#### 1. 数据文件缺失
```
错误: FileNotFoundError: [Errno 2] No such file or directory: 'data/test0.csv'
解决: 运行 `python reat_dataset.py` 生成数据文件
```

#### 2. 内存不足
```
错误: MemoryError during training
解决: 减少NUM_EXPERIMENTS或NUM_CLIENT参数
```

#### 3. 导入模块失败
```
错误: ModuleNotFoundError: No module named 'distance'
解决: 确保在项目根目录运行，检查Python路径
```

#### 4. 客户端模型不存在
```
错误: 客户端模型文件不存在
解决: 让程序自动生成，或手动运行 `generate_client_model()`
```

### 调试模式

开启详细日志输出：

```python
# 在run.py中设置
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 性能优化建议

### 1. 硬件优化
- **GPU加速**: 使用CUDA支持的PyTorch版本
- **内存**: 至少8GB内存用于大规模实验
- **存储**: SSD提升文件I/O性能

### 2. 参数调优
```python
# 高性能配置
NUM_CLIENT = 4              # 减少客户端数量
NUM_EXPERIMENTS = 20        # 减少实验次数
HIDDEN_DIM = 5              # 减少模型复杂度
MODEL_NAME = "MLP_2"        # 使用较小的模型
```

### 3. 批量运行
```bash
# 使用后台运行
nohup python run.py > experiment.log 2>&1 &

# Windows下后台运行
start /B python run.py > experiment.log 2>&1
```

## 🤝 贡献指南

我们欢迎社区贡献！请遵循以下步骤：

### 提交问题
1. 检查[Issues](https://github.com/yourusername/federated-battery-classification/issues)是否已存在相关问题
2. 使用Issue模板提供详细信息
3. 包含错误日志和系统环境信息

### 提交代码
1. Fork项目到你的GitHub
2. 创建新分支: `git checkout -b feature/your-feature`
3. 提交更改: `git commit -m 'Add your feature'`
4. 推送分支: `git push origin feature/your-feature`
5. 创建Pull Request

### 代码规范
- 使用PEP 8代码风格
- 添加必要的注释和文档字符串
- 确保向后兼容性
- 添加单元测试

## 📄 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件。

## 📞 联系方式
- **邮箱**: h-hu24@mails.tsinghua.edu.cn



## 📊 项目统计

![GitHub stars](https://img.shields.io/github/stars/yourusername/federated-battery-classification?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/federated-battery-classification?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/federated-battery-classification)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/federated-battery-classification)

---

⭐ 如果这个项目对你有帮助，请给它一个Star！