# 🚀 快速开始指南

这个指南帮助你在5分钟内运行联邦学习电池分类系统。

## ⚡ 5分钟快速体验

### 1. 检查Python环境

```bash
# 确保Python 3.7+已安装
python --version
```

### 2. 安装依赖

```bash
# 方法一：快速安装（推荐）
pip install pandas numpy scikit-learn torch matplotlib seaborn scipy

# 方法二：使用requirements.txt
pip install -r requirements.txt
```

### 3. 运行快速演示

```bash
# 运行交互式快速开始
python example_usage.py
```

这将：
- 生成4个客户端模型
- 创建可视化图表
- 展示基本联邦学习流程

### 4. 运行完整实验（可选）

```bash
# 运行完整的联邦学习实验
python run.py
```

## 🔍 验证安装

运行验证脚本检查系统：

```bash
python test_run.py
```

预期输出：
```
run.py 功能验证测试
==================================================
[OK] 所有核心功能验证通过
```

## 📁 检查输出文件

成功运行后，你应该看到以下文件：

```
├── client_model/42/          # 客户端模型
├── data/                     # 数据文件（如果有）
└── *.png                    # 生成的图表
```

## ❓ 遇到问题？

1. **数据文件缺失**：运行 `python reat_dataset.py`
2. **内存不足**：减少实验次数到10次
3. **导入错误**：确保在项目根目录运行

## 📞 需要帮助？

- 查看 [README.md](README.md) 获取详细文档
- 提交 [Issue](https://github.com/yourusername/federated-battery-classification/issues)

---

🎉 **恭喜！你已经成功运行了联邦学习电池分类系统！**