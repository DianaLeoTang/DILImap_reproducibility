# DILImap Reproducibility Notebooks
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17290520.svg)](https://doi.org/10.5281/zenodo.17290520)

This repository contains Jupyter notebooks supporting the reproducibility of all results in the DILImap project. 
- DILImap codebase: [github.com/Cellarity/DILImap](https://github.com/Cellarity/DILImap)
- DILImap website: [dilimap.org](https://dilimap.org/)

## 🚀 Quick Start

### Installing DILImap Package

The `dilimap` package is a standalone Python package that has been published to PyPI. You have several installation options:

#### Option 1: Install from PyPI (Original Version)
```bash
pip install dilimap
```

#### Option 2: Install from Your Forked Repository
If you have forked the DILImap repository to your own GitHub, you can install your version:

```bash
# Replace YOUR_USERNAME with your GitHub username
pip install git+https://github.com/YOUR_USERNAME/DILImap.git

# Or for editable installation (recommended for development)
git clone https://github.com/YOUR_USERNAME/DILImap.git
cd DILImap
pip install -e .
```

#### Option 3: Install from Local Source
If you have the DILImap source code locally:

```bash
cd /path/to/DILImap
pip install -e .
```

**Note**: You only need to install ONE version. If you've forked the repository and made modifications, use Option 2 to install your fork version. The package name remains `dilimap` regardless of which source you install from.

> 📖 **For detailed instructions on installing from a forked repository**, see [INSTALLATION_FORK.md](INSTALLATION_FORK.md)

### Verify Installation
```python
import dilimap as dmap
dmap.logging.print_version()
```

## 📁 Repository Structure

### 📚 Tutorials

This directory contains hands-on, end-to-end tutorials designed to help new users apply the DILImap framework.
These are ideal starting points for users who want to apply the pipeline to their own data or explore the model outputs.
- `1_Compute_Pathway_Signatures.ipynb`
- `2_Run_ToxPredictor_Model.ipynb`


### 📓 Reproducibility
This directory contains the full set of notebooks used to generate the results shown in the DILImap publication:

#### Data Preparation
- `1.1_DataPrep_DILI_Labels.ipynb`
- `1.2_DataPrep_Cmax_Values.ipynb`
- `1.3_DataPrep_Viability_Assay.ipynb`

#### Model Training
- `2.1_Training_Gene_Signatures.ipynb`
- `2.2_Training_Pathway_Signatures.ipynb`
- `2.3_Training_ToxPredictor_Model.ipynb`

#### Model Validation
- `3.1_Validation_Gene_Signatures.ipynb`
- `3.2_Validation_Pathway_Signatures.ipynb`
- `3.3_Validation_ToxPredictor_Model.ipynb`

#### Results & Benchmarking
- `4.1_Results_Main_Figures.ipynb`
- `4.2_Benchmarking_Insilico_Models.ipynb`
- `4.3_Benchmarking_Invitro_Models.ipynb`