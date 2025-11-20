# 故障排除指南

本文档记录运行 DILImap 可重现性 notebooks 时可能遇到的常见错误及其解决方案。

## 🔧 常见错误及解决方案

### 1. ValueError: The truth value of a Index is ambiguous

**错误信息**：
```
ValueError: The truth value of a Index is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
```

**原因**：
在较新版本的 pandas 中，直接对 Index 对象进行布尔判断（如 `value in df.index`）可能会导致歧义错误。

**解决方案**：
将 Index 转换为列表后再进行判断：

```python
# ❌ 错误写法
if value not in df.index:
    ...

# ✅ 正确写法
if value not in df.index.tolist():
    # 或者
if value not in list(df.index):
    ...
```

**已修复位置**：
- `reproducibility/1.1_DataPrep_DILI_Labels.ipynb` - Cell 36

---

### 2. FutureWarning: Setting an item of incompatible dtype

**错误信息**：
```
FutureWarning: Setting an item of incompatible dtype is deprecated and will raise an error in a future version of pandas.
```

**原因**：
尝试将不兼容的数据类型赋值给 DataFrame 列。

**解决方案**：
在赋值前显式转换数据类型：

```python
# ❌ 可能导致警告
df['column'] = 'string_value'  # 如果 column 是 float64 类型

# ✅ 正确写法
df['column'] = df['column'].astype(str)
df.loc[index, 'column'] = 'string_value'
```

---

### 3. ImportError: No module named 'dilimap'

**错误信息**：
```
ModuleNotFoundError: No module named 'dilimap'
```

**原因**：
`dilimap` 包未安装或未在正确的环境中安装。

**解决方案**：

1. **确认环境激活**：
   ```bash
   conda activate dilimap
   # 或
   source activate dilimap
   ```

2. **安装 dilimap 包**：
   ```bash
   # 从 PyPI 安装
   pip install dilimap
   
   # 或从你的 fork 安装
   pip install git+https://github.com/DianaLeoTang/DILImap.git
   ```

3. **验证安装**：
   ```python
   import dilimap as dmap
   dmap.logging.print_version()
   ```

详细安装说明请参考 [INSTALLATION_FORK.md](INSTALLATION_FORK.md)

---

### 4. S3 访问错误（专有数据）

**错误信息**：
```
ClientError: An error occurred (NoSuchVersion) when calling the GetObject operation: 
The specified version does not exist.
Package: s3://dilimap/proprietary/data. Top hash: ...
```

**原因**：
- **专有数据访问限制**：训练数据文件是专有的，受知识产权限制保护
- S3 凭证未配置或权限不足
- 网络连接问题
- 数据包版本不存在或已更新

**解决方案**：

1. **申请数据访问权限**（推荐）：
   - 这些数据需要数据共享协议才能访问
   - 联系邮箱：**DILImap@cellarity.com**
   - 或访问项目 GitHub Issues 页面申请访问权限

2. **检查访问权限**（如果您已有权限）：
   - 确认已签署数据共享协议
   - 检查 S3 凭证是否正确配置
   - 验证网络连接是否正常
   - 确认数据包版本是否匹配

3. **使用本地数据**：
   - 如果您已有本地数据文件，代码会自动尝试从以下路径加载：
     - `training_data_counts.h5ad`
     - `../data/training_data_counts.h5ad`
     - `./data/training_data_counts.h5ad`
   - 或手动指定路径：
     ```python
     import anndata as ad
     adata = ad.read_h5ad('path/to/your/local/training_data_counts.h5ad')
     ```

4. **错误处理**：
   - 新版本的 notebook 已包含自动错误处理和说明
   - 如果遇到错误，会显示详细的解决方案和联系信息

**注意**：`2.1_Training_Gene_Signatures.ipynb` 等训练相关的 notebook 需要访问专有数据。如果您只是想使用预训练模型进行预测，可以跳过训练步骤，直接使用验证和结果分析的 notebook。

---

### 4.1 AWS 凭证错误

**错误信息**：
```
AWS credentials not found in environment or .env file.
That didn't work. Unable to locate credentials. Try again.
```

**原因**：
- AWS 凭证未配置
- 某些 S3 操作需要凭证（即使访问公开数据）

**解决方案**：

1. **对于公开数据**（推荐）：
   - 通常不需要配置凭证
   - 代码会自动处理凭证错误并继续运行
   - 如果遇到此错误，可以忽略并继续

2. **如果需要访问专有数据**：
   - 需要配置 AWS 凭证
   - **方法 1**：环境变量
     ```bash
     export AWS_ACCESS_KEY_ID=your_access_key
     export AWS_SECRET_ACCESS_KEY=your_secret_key
     ```
   - **方法 2**：AWS CLI
     ```bash
     pip install awscli
     aws configure
     ```
   - **方法 3**：创建凭证文件
     创建 `~/.aws/credentials` 文件：
     ```ini
     [default]
     aws_access_key_id = your_access_key
     aws_secret_access_key = your_secret_key
     ```

3. **获取凭证**：
   - 如果您有数据访问权限，项目维护者会提供凭证
   - 联系：DILImap@cellarity.com

**已修复的 Notebook**：
- ✅ `4.1_Results_Main_Figures.ipynb` - 已添加自动错误处理

---

### 4.2 gseapy 版本兼容性错误

**错误信息**：
```
enrichr() got an unexpected keyword argument 'organism'
Failed after 5 attempts for index X.
```

**原因**：
- gseapy 版本更新后，`enrichr()` 函数的 API 发生了变化
- 新版本可能不再支持 `organism` 参数
- 导致 `pathway_signatures` 函数执行失败

**解决方案**：

1. **尝试从 S3 加载预计算的结果**（推荐）：
   - 代码已自动尝试从 S3 加载 `validation_data_pathways.h5ad`
   - 如果可用，会直接使用预计算的数据

2. **更新或降级 gseapy**：
   ```bash
   # 尝试更新到最新版本
   pip install --upgrade gseapy
   
   # 或安装特定版本（如果知道兼容版本）
   pip install gseapy==0.10.8  # 示例版本号
   ```

3. **检查 gseapy 版本**：
   ```python
   import gseapy
   print(gseapy.__version__)
   ```

4. **使用预计算数据**：
   - 如果计算失败，可以尝试从 S3 直接加载预计算的结果
   - 代码已包含自动回退机制

**已修复的 Notebook**：
- ✅ `3.2_Validation_Pathway_Signatures.ipynb` - 已添加错误处理和回退方案

---

### 5. Docker 相关错误（DESeq2 分析）

**错误信息**：
```
Docker is installed but cannot connect to the Docker daemon.
Falling back to precomputed DESeq2 data...
```

**原因**：
- Docker 未安装
- Docker daemon 未运行

**解决方案**：

1. **启动 Docker**：
   ```bash
   # macOS/Linux
   sudo systemctl start docker
   # 或启动 Docker Desktop 应用
   ```

2. **使用预计算数据**：
   如果 Docker 不可用，代码会自动回退到使用预计算的 DESeq2 数据，这通常可以正常工作。

---

### 6. 依赖包版本冲突

**错误信息**：
```
ImportError: cannot import name 'xxx' from 'yyy'
AttributeError: module 'xxx' has no attribute 'yyy'
```

**原因**：
依赖包版本不兼容。

**解决方案**：

1. **检查已安装版本**：
   ```bash
   pip list | grep pandas
   pip list | grep numpy
   ```

2. **安装兼容版本**：
   ```bash
   pip install pandas>=1.5.0 numpy>=1.23.0
   ```

3. **使用虚拟环境**：
   ```bash
   conda create -n dilimap python=3.10
   conda activate dilimap
   pip install -r requirements.txt  # 如果有的话
   ```

---

### 7. 文件路径错误

**错误信息**：
```
FileNotFoundError: [Errno 2] No such file or directory: '../data/xxx.csv'
```

**原因**：
- 数据文件未下载
- 路径不正确
- 工作目录不对

**解决方案**：

1. **检查工作目录**：
   ```python
   import os
   print(os.getcwd())  # 确认当前目录
   ```

2. **创建数据目录**：
   ```bash
   mkdir -p ../data
   ```

3. **下载必要数据**：
   按照 notebook 中的说明下载数据文件

---

## 🔍 调试技巧

### 1. 启用详细输出

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. 检查数据形状

```python
print(df.shape)
print(df.head())
print(df.columns)
print(df.index)
```

### 3. 使用 try-except 捕获错误

```python
try:
    # 可能出错的代码
    result = some_function()
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
```

### 4. 验证数据类型

```python
print(df.dtypes)
print(df.info())
```

---

## 📞 获取帮助

如果遇到其他问题：

1. **查看项目文档**：
   - [README.md](README.md)
   - [INSTALLATION_FORK.md](INSTALLATION_FORK.md)
   - [project_info.md](project_info.md)

2. **检查 GitHub Issues**：
   - 原项目：[Cellarity/DILImap](https://github.com/Cellarity/DILImap/issues)
   - 可重现性项目：[Cellarity/DILImap_reproducibility](https://github.com/Cellarity/DILImap_reproducibility/issues)

3. **联系支持**：
   - DILImap@cellarity.com

---

**最后更新**: 2025年

