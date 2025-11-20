# 使用 Fork 版本安装 DILImap

本文档说明如何安装和使用你 fork 的 DILImap 版本。

## 📦 关于 DILImap 包

`dilimap` 是一个独立的 Python 包，已经发布到 PyPI。如果你 fork 了原仓库并进行了修改，应该安装你自己的 fork 版本而不是原作者的版本。

## 🔧 安装 Fork 版本

### 方法 1：直接从 GitHub 安装（推荐）

```bash
# 安装你的 fork 版本
pip install git+https://github.com/DianaLeoTang/DILImap.git

# 如果需要安装特定分支
pip install git+https://github.com/DianaLeoTang/DILImap.git@branch-name
```

### 方法 2：可编辑模式安装（开发推荐）

如果你需要修改代码并实时看到效果，使用可编辑模式安装：

```bash
# 1. 克隆你的 fork 仓库
git clone https://github.com/DianaLeoTang/DILImap.git
cd DILImap

# 2. 以可编辑模式安装
pip install -e .

# 或者使用 conda 环境
conda activate dilimap
pip install -e .
```

**可编辑模式的优势**：
- 修改代码后无需重新安装
- 可以直接在代码库中调试
- 适合开发和测试

### 方法 3：从本地路径安装

如果你已经将代码下载到本地：

```bash
cd /path/to/your/DILImap
pip install -e .
```

## ✅ 验证安装

安装完成后，验证是否正确安装：

```python
import dilimap as dmap
dmap.logging.print_version()
```

应该会显示类似以下信息：
```
Running dilimap X.X.X (python 3.10.16) on YYYY-MM-DD HH:MM.
```

## 🔄 更新 Fork 版本

### 更新到最新版本

```bash
# 如果使用可编辑模式安装
cd /path/to/your/DILImap
git pull origin main  # 或你的主分支名
# 代码会自动更新，无需重新安装

# 如果直接从 GitHub 安装
pip install --upgrade git+https://github.com/DianaLeoTang/DILImap.git
```

### 同步原仓库的更新

如果你想将原仓库的更新合并到你的 fork：

```bash
cd /path/to/your/DILImap

# 添加上游仓库（只需执行一次）
git remote add upstream https://github.com/Cellarity/DILImap.git

# 获取上游更新
git fetch upstream

# 合并到你的分支
git merge upstream/main

# 推送到你的 fork
git push origin main
```

## 📝 重要提示

1. **只需安装一个版本**：如果你已经 fork 并修改了代码，只需要安装你的 fork 版本，不需要安装原作者的版本。

2. **包名不变**：无论从哪个源安装，包名都是 `dilimap`，导入方式不变：
   ```python
   import dilimap as dmap
   ```

3. **版本冲突**：如果之前安装了原作者的版本，先卸载再安装你的版本：
   ```bash
   pip uninstall dilimap
   pip install git+https://github.com/DianaLeoTang/DILImap.git
   ```

4. **开发环境**：如果需要在 fork 版本上开发，推荐使用可编辑模式安装（`pip install -e .`），这样修改代码后可以立即生效。

## 🐛 常见问题

### Q: 安装后导入失败怎么办？

**A**: 检查以下几点：
1. 确认安装成功：`pip list | grep dilimap`
2. 检查 Python 环境：确保在正确的 conda/virtualenv 环境中
3. 重新安装：`pip uninstall dilimap && pip install git+https://github.com/DianaLeoTang/DILImap.git`

### Q: 如何查看当前安装的版本来源？

**A**: 
```bash
pip show dilimap
```
会显示包的安装位置和版本信息。

### Q: 可以同时安装多个版本吗？

**A**: 不可以。Python 包管理器只允许安装一个版本的包。如果需要测试不同版本，使用不同的虚拟环境。

## 🔗 相关链接

- **你的 Fork 仓库**：[https://github.com/DianaLeoTang/DILImap](https://github.com/DianaLeoTang/DILImap)
- **原仓库**：[https://github.com/Cellarity/DILImap](https://github.com/Cellarity/DILImap)
- **PyPI 原版本**：`pip install dilimap`

---

**最后更新**: 2025年

