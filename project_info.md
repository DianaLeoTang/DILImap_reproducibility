# DILImap 可重现性项目

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17290520.svg)](https://doi.org/10.5281/zenodo.17290520)

> **DILImap Reproducibility Notebooks** - 基于转录组学数据的药物性肝损伤（DILI）预测框架的完整可重现性仓库

## 📋 项目简介

本项目提供了 DILImap 论文中所有结果的完整可重现性支持。DILImap 是一个基于转录组学数据预测药物性肝损伤（Drug-Induced Liver Injury, DILI）的机器学习框架，通过分析基因表达和通路激活模式来评估化合物的肝毒性风险。

### 核心特性

- 🧬 **转录组学分析**：使用 DESeq2 进行差异表达分析，WikiPathways 进行通路富集
- 🤖 **多模型架构**：基因特征、通路特征和集成 ToxPredictor 模型
- 📊 **完整基准测试**：与 DILIGeNN、DILIPredictor、TxGemma 等先进模型对比
- 🔬 **可重现性**：提供完整的 Jupyter notebooks 和数据流程
- 📚 **教程支持**：包含新手友好的端到端教程

### 相关链接

- **DILImap 主代码库**：[github.com/Cellarity/DILImap](https://github.com/Cellarity/DILImap)
- **DILImap 网站**：[dilimap.org](https://dilimap.org/)
- **论文 DOI**：[10.5281/zenodo.17290520](https://doi.org/10.5281/zenodo.17290520)

---

## 🚀 快速开始

### 前置要求

- Python 3.10 或更高版本
- Conda 或 Miniconda
- Jupyter Notebook 或 JupyterLab
- Git

### 安装步骤

#### 1. 克隆仓库

```bash
git clone https://github.com/Cellarity/DILImap_reproducibility.git
cd DILImap_reproducibility
```

#### 2. 创建 Conda 环境

```bash
# 创建基础环境
conda create -n dilimap python=3.10 -y
conda activate dilimap
```

#### 3. 安装 DILImap 包

本项目需要安装 `dilimap` Python 包。如果你已经 fork 了 DILImap 仓库，请安装你的 fork 版本：

```bash
# 方法1：直接从你的 GitHub fork 安装（推荐）
pip install git+https://github.com/DianaLeoTang/DILImap.git

# 方法2：克隆后以可编辑模式安装（适合开发调试）
git clone https://github.com/DianaLeoTang/DILImap.git
cd DILImap
pip install -e .
cd ..
```

**注意**：
- 如果你 fork 了仓库并进行了修改，应该使用你的 fork 版本而不是原作者的版本
- 如果使用可编辑模式安装（`pip install -e .`），修改代码后无需重新安装即可生效
- 如果之前安装了原作者的版本，先卸载：`pip uninstall dilimap`

> 📖 详细的 fork 版本安装说明请参考 [INSTALLATION_FORK.md](INSTALLATION_FORK.md)

#### 4. 安装项目依赖

```bash
# 安装 Python 依赖
pip install pandas numpy matplotlib seaborn anndata

# 安装生物信息学工具（可选，用于通路分析）
conda install -c bioconda gseapy -y

# 安装 Jupyter
pip install jupyter jupyterlab
```

#### 5. 启动 Jupyter

```bash
jupyter lab
# 或
jupyter notebook
```

---

## 📁 项目结构

```
DILImap_reproducibility/
│
├── README.md                          # 本文件
├── LICENSE                            # 许可证文件
├── pyproject.toml                     # Python 项目配置
│
├── tutorials/                         # 📚 教程目录（新手入门）
│   ├── 1_Compute_Pathway_Signatures.ipynb      # 教程1：计算通路特征
│   └── 2_Run_ToxPredictor_Model.ipynb         # 教程2：运行 ToxPredictor 模型
│
└── reproducibility/                   # 📓 完整可重现性流程
    │
    ├── README.md                      # 可重现性说明
    │
    ├── 1.1_DataPrep_DILI_Labels.ipynb         # 数据准备：DILI 标签
    ├── 1.2_DataPrep_Cmax_Values.ipynb         # 数据准备：Cmax 值
    ├── 1.3_DataPrep_Viability_Assay.ipynb     # 数据准备：细胞活力实验
    │
    ├── 2.1_Training_Gene_Signatures.ipynb     # 模型训练：基因特征
    ├── 2.2_Training_Pathway_Signatures.ipynb  # 模型训练：通路特征
    ├── 2.3_Training_ToxPredictor_Model.ipynb  # 模型训练：ToxPredictor
    │
    ├── 3.1_Validation_Gene_Signatures.ipynb    # 模型验证：基因特征
    ├── 3.2_Validation_Pathway_Signatures.ipynb # 模型验证：通路特征
    ├── 3.3_Validation_ToxPredictor_Model.ipynb # 模型验证：ToxPredictor
    │
    ├── 4.1_Results_Main_Figures.ipynb          # 结果：主要图表
    ├── 4.2_Benchmarking_Insilico_Models.ipynb  # 基准测试：计算模型
    └── 4.3_Benchmarking_Invitro_Models.ipynb   # 基准测试：体外模型
    │
    └── insilico_benchmarks/           # 🔬 计算模型基准测试
        │
        ├── README.md                   # 基准测试详细说明
        ├── __init__.py
        │
        ├── run_diligenn.py            # DILIGeNN 预测脚本
        ├── run_dilipredictor.py       # DILIPredictor 预测脚本
        ├── run_txgemma_dili.py        # TxGemma 预测脚本
        ├── graph_gen_diligenn.py      # DILIGeNN 图生成脚本
        │
        ├── diligenn_models/           # DILIGeNN 预训练模型
        │   └── dlst/
        │       ├── initial_model_p20_20/      # 初始模型权重
        │       └── warm_starts_p50/           # 热启动模型权重
        │
        ├── dilipr_environment.yaml    # DILIPredictor Conda 环境配置
        │
        ├── smiles_data/              # SMILES 数据文件
        │   ├── DILIGeNN_seen_1167.csv
        │   ├── DILIPredictor_seen_1111.csv
        │   ├── txgemma_train.csv
        │   ├── unseen_smiles_diligenn_349.csv
        │   ├── unseen_smiles_dilipr_483.csv
        │   └── unseen_smiles_txgemma_715.csv
        │
        └── predictions/              # 预测结果输出
            ├── benchmark_diligenn_331_predictions.csv
            ├── benchmark_dilipr_471_predictions.csv
            └── benchmark_txgemma_all_variants_715_predictions.csv
```

---

## 📖 使用指南

### 新手入门

如果你是第一次使用 DILImap，建议按以下顺序学习：

1. **从教程开始**：运行 `tutorials/` 目录下的两个教程 notebook
   - `1_Compute_Pathway_Signatures.ipynb`：学习如何计算通路特征
   - `2_Run_ToxPredictor_Model.ipynb`：学习如何运行预测模型

2. **理解数据流程**：
   ```
   原始转录组数据 → 数据预处理 → 差异表达分析 → 通路富集 → 特征提取 → 模型训练 → 预测
   ```

### 完整可重现性流程

要重现论文中的所有结果，按以下顺序执行 notebooks：

#### 阶段 1：数据准备（1.x）

```bash
# 在 Jupyter 中依次运行：
1.1_DataPrep_DILI_Labels.ipynb      # 准备 DILI 标签数据
1.2_DataPrep_Cmax_Values.ipynb      # 准备 Cmax 值数据
1.3_DataPrep_Viability_Assay.ipynb  # 准备细胞活力实验数据
```

#### 阶段 2：模型训练（2.x）

```bash
2.1_Training_Gene_Signatures.ipynb      # 训练基因特征模型
2.2_Training_Pathway_Signatures.ipynb  # 训练通路特征模型
2.3_Training_ToxPredictor_Model.ipynb  # 训练集成 ToxPredictor 模型
```

#### 阶段 3：模型验证（3.x）

```bash
3.1_Validation_Gene_Signatures.ipynb      # 验证基因特征模型
3.2_Validation_Pathway_Signatures.ipynb  # 验证通路特征模型
3.3_Validation_ToxPredictor_Model.ipynb  # 验证 ToxPredictor 模型
```

#### 阶段 4：结果分析（4.x）

```bash
4.1_Results_Main_Figures.ipynb          # 生成主要结果图表
4.2_Benchmarking_Insilico_Models.ipynb  # 与计算模型对比
4.3_Benchmarking_Invitro_Models.ipynb   # 与体外模型对比
```

### 运行基准测试

#### DILIGeNN（图神经网络）

```bash
cd reproducibility/insilico_benchmarks

# 方法1：使用 Python 脚本
python run_diligenn.py \
    --input smiles_data/unseen_smiles_diligenn_349.csv \
    --output predictions/benchmark_diligenn_predictions.csv

# 方法2：在 Python 中调用
python
>>> from insilico_benchmarks import diligenn_predict_outer_folds_warm_starts
>>> import pandas as pd
>>> df = pd.read_csv('smiles_data/unseen_smiles_diligenn_349.csv')
>>> results = diligenn_predict_outer_folds_warm_starts(df)
```

**依赖要求**：
```bash
pip install torch torch-geometric rdkit captum networkx scikit-learn
```

#### DILIPredictor（随机森林集成）

```bash
# 1. 设置 DILIPredictor 环境
conda env create -f dilipr_environment.yaml
conda activate dilipr-conda
pip install scikit-learn==1.2.0

# 2. 克隆 DILIPredictor 仓库
git clone https://github.com/Manas02/dili-pip.git

# 3. 运行预测
python run_dilipredictor.py \
    -i smiles_data/unseen_smiles_dilipr_483.csv \
    -o predictions/benchmark_dilipr_predictions.csv \
    --dili_pip_repo /path/to/dili-pip
```

#### TxGemma（大语言模型）

```bash
# 需要 Google Cloud Platform 和 Vertex AI 配置
python run_txgemma_dili.py \
    --project_id YOUR_GCP_PROJECT_ID \
    --endpoint_id YOUR_ENDPOINT_ID \
    --endpoint_region us-central1 \
    --input smiles_data/unseen_smiles_txgemma_715.csv \
    --output predictions/benchmark_txgemma_predictions.csv
```

**依赖要求**：
- Google Cloud SDK
- Vertex AI API 访问权限
- 服务账户凭证

---

## 🔧 环境配置详解

### Python 环境

推荐使用 Conda 管理环境：

```bash
# 创建新环境
conda create -n dilimap python=3.10 -y
conda activate dilimap

# 安装基础依赖
pip install dilimap pandas numpy matplotlib seaborn anndata jupyter
```

### 生物信息学工具

```bash
# 安装 DESeq2 相关工具（R 环境）
conda install -c bioconda r-deseq2 r-base -y

# 安装通路分析工具
conda install -c bioconda gseapy -y
```

### 深度学习环境（用于 DILIGeNN）

```bash
# 安装 PyTorch（根据你的 CUDA 版本选择）
# CPU 版本
pip install torch torchvision torchaudio

# CUDA 11.8 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 PyTorch Geometric
pip install torch-geometric

# 安装 RDKit
conda install -c conda-forge rdkit -y
```

### 验证安装

```python
# 在 Python 中验证
import dilimap as dmap
import torch
import torch_geometric
from rdkit import Chem

print(f"DILImap version: {dmap.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"RDKit version: {Chem.__version__}")
```

---

## 📊 数据流程

```
┌─────────────────────────────────────────────────────────────┐
│                    原始转录组数据                              │
│              (RNA-seq 计数矩阵)                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   数据预处理                                  │
│  • 质量控制                                                  │
│  • 标准化                                                    │
│  • 批次校正                                                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              差异表达分析 (DESeq2)                            │
│  • 计算 log₂ 折叠变化                                         │
│  • 计算调整后的 p 值                                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           通路富集分析 (WikiPathways)                          │
│  • 计算通路激活分数 (–log₁₀ p 值)                            │
│  • 提取通路特征                                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   特征提取                                    │
│  • 基因特征：差异表达基因                                     │
│  • 通路特征：通路激活分数                                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 模型训练与验证                                 │
│  • 基因特征模型                                              │
│  • 通路特征模型                                              │
│  • ToxPredictor 集成模型                                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   预测与评估                                  │
│  • DILI 风险预测                                             │
│  • 性能评估（AUROC, 敏感性, 特异性等）                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 核心功能模块

### 1. 转录组学分析

- **差异表达分析**：使用 DESeq2 识别处理组与对照组的差异表达基因
- **通路富集**：使用 WikiPathways 基因集进行通路富集分析
- **特征提取**：从差异表达和通路激活中提取预测特征

### 2. 模型架构

- **基因特征模型**：基于差异表达基因的预测模型
- **通路特征模型**：基于通路激活分数的预测模型
- **ToxPredictor**：集成基因和通路特征的最终预测模型

### 3. 基准测试框架

- **公平对比**：确保测试集化合物在训练时未见
- **多模型评估**：与 DILIGeNN、DILIPredictor、TxGemma 对比
- **全面指标**：平衡准确率、敏感性、特异性、AUROC

---

## ❓ 常见问题

### Q1: 如何在自己的数据上使用 DILImap？

**A**: 参考 `tutorials/` 目录下的教程 notebooks，它们展示了如何：
1. 准备输入数据（转录组计数矩阵）
2. 计算通路特征
3. 运行 ToxPredictor 模型进行预测

### Q2: 需要多少计算资源？

**A**: 
- **基础分析**：8GB RAM，4 核 CPU 即可
- **完整重现**：16GB RAM，8 核 CPU 推荐
- **DILIGeNN 训练**：需要 GPU（推荐 8GB+ 显存）

### Q3: 如何安装 DILImap 包？

**A**: 
```bash
# 从 PyPI 安装
pip install dilimap

# 或从源码安装
git clone https://github.com/Cellarity/DILImap.git
cd DILImap
pip install -e .
```

### Q4: 基准测试脚本运行失败怎么办？

**A**: 
1. 检查依赖是否完整安装
2. 确认数据文件路径正确
3. 查看各基准测试目录下的 README.md 获取详细说明
4. 检查环境变量和配置文件

### Q5: 如何引用本项目？

**A**: 请使用以下格式：
```bibtex
@software{dilimap_reproducibility,
  title = {DILImap Reproducibility Notebooks},
  author = {...},
  doi = {10.5281/zenodo.17290520},
  url = {https://github.com/Cellarity/DILImap_reproducibility}
}
```

---

## 📝 依赖清单

### 核心依赖

```
dilimap>=1.0.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.5.0
seaborn>=0.12.0
anndata>=0.8.0
jupyter>=1.0.0
```

### 生物信息学工具

```
gseapy>=1.0.0
r-deseq2>=1.36.0
```

### 深度学习（DILIGeNN）

```
torch>=1.12.0
torch-geometric>=2.0.0
rdkit>=2023.9.2
captum>=0.6.0
networkx>=3.0
scikit-learn>=1.2.0
```

### 传统机器学习（DILIPredictor）

```
scikit-learn==1.2.0  # 特定版本要求
mordred>=1.2.0
shap>=0.41.0
dimorphite-dl>=1.2.5
```

---

## 🔗 相关资源

- [DILImap 主项目](https://github.com/Cellarity/DILImap)
- [DILImap 网站](https://dilimap.org/)
- [DILIGeNN 原始论文](https://github.com/tlee23-ic/GNN_DILI)
- [DILIPredictor 原始论文](https://github.com/Manas02/dili-pip)
- [TxGemma 项目](https://github.com/google-gemini/gemma-cookbook/tree/main/TxGemma)