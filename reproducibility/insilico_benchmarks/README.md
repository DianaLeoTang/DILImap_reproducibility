# In-Silico DILI Benchmarks

This folder contains **reproducible scripts** and resources for benchmarking state-of-the-art in-silico Drug-Induced Liver Injury (DILI) prediction models against the DILImap and ToxPredictor model. The benchmarking framework ensures fair comparison by evaluating models on compounds that are **unseen** during their respective training phases.

# 📋 Overview

This benchmarking suite evaluates four different DILI prediction models including state-of-the-art approaches:
- **DILIGeNN**: Graph neural networks for molecular representation - **1,167 training compounds (DILIRank / DILIst)**
- **DILIPredictor**: Traditional machine learning with molecular descriptors - **1,111 training compounds (DILIRank / DILIst)**
- **TxGemma**: Large language models from Google DeepMind - **379 train + 96 test compounds (TDC)**
- **ToxPredictor**: Transcriptomics-based DILI prediction  - **240 train + 60  test (DILIMap)**


Results generate **Figure 6, Supplementary Fig S10, Table S4** in the DILImap manuscript and provide comprehensive performance metrics across different model architectures.

## 🗂️ Directory Structure

```
insilico_benchmarks/
├── README.md                    # This documentation
├── __init__.py                  # Module initialization
│
├── REPRODUCIBILITY SCRIPTS
├── run_diligenn.py              # DILIGeNN predictions 
├── run_dilipredictor.py         # DILIPredictor predictions 
├── run_txgemma_dili.py          # TxGemma predictions 
├── graph_gen_diligenn.py        # DILIGeNN preprocessing 
│
├── MODEL WEIGHTS & CONFIGS
├── diligenn_models/             # Pre-trained DILIGeNN weights
├── dilipr_environment.yaml      # DILIPredictor conda environment
│
├── DATA & RESULTS
├── smiles_data/                 # Input datasets for each model
└── predictions/                 # Benchmark prediction outputs
```


# 📊 BENCHMARK FRAMEWORK

This benchmarking framework evaluates three state-of-the-art DILI prediction models against compounds that are **unseen** during their respective training phases, ensuring fair comparison and avoiding data leakage.

### Data Processing Pipeline
1. **Overlap Detection**: InChiKey14, SMILES, and name-based matching to identify training/test overlaps
2. **Molecular Standardization**: RDKit-based standardization for consistency across models
3. **Unseen Compound Filtering**: Removal of compounds present in original training sets
4. **Cross-Validation**: Stratified k-fold validation with fixed random seeds for reproducibility

### Unseen Compound Evaluation
- **DILIGeNN**: 331 unseen compounds (after removing training overlap)
- **DILIPredictor**: 471 unseen compounds  
- **TxGemma**: 715 unseen compounds

### Overlap with DILImap Transcriptomics Dataset 
- **DILIGeNN Unseen ∩ DILImap**: 8 compounds
- **DILIPredictor Unseen ∩ DILImap**: 30 compounds
- **TxGemma Unseen ∩ DILImap**: 97 compounds
#### Among chemistry based methods
- **DILIPredictor Unseen ∩ DILIGeNN Unseen**: 314 compounds
- **TxGemma Unseen ∩ DILIPredictor Unseen ∩ DILIGeNN Unseen**: 303 compounds

### Evaluation Metrics
- **Balanced Accuracy**: Accounts for class imbalance across datasets
- **Specificity**: True negative rate (No DILI identification)
- **Sensitivity**: True positive rate (DILI identification)
- **AUROC**: Area under ROC curve 

## Important Caveats
- **Variable Test set composition**: Class imbalance and chemical structure diversity vary significantly across models and overlaps
- **Limited Overlap**: Small intersection sizes with DILImap may affect statistical power
- **Environment Dependencies**: Each model requires specific software environments and versions
- **Dataset Size Disparity**: Training set sizes vary dramatically (300-1,167 compounds) affecting model complexity


# BENCHMARK MODELS


## 1. DILIGeNN (Graph Neural Network)

### Original Work
```
Lee T, Posma J. Improving Drug-Induced Liver Injury Prediction Using Graph Neural Networks with Augmented Graph Features from Molecular Optimisation. ChemRxiv. 2025; doi:10.26434/chemrxiv-2024-d12gk-v2 (pre-print).
```
**GitHub**:  [https://github.com/tlee23-ic/GNN_DILI/tree/main]

### Model Description
DILIGeNN employs graph neural networks to learn molecular representations directly from chemical structure graphs. It uses GraphSAGE architecture with warm-start training for enhanced performance.

### Reproducibility Scripts
- **Main Script**: `run_diligenn.py` 
- **Preprocessing**: `graph_gen_diligenn.py` 

### Key Functions & Classes
```python
# Main prediction functions
diligenn_predict_outer_folds()              # Base model predictions
diligenn_predict_outer_folds_warm_starts()  # Enhanced warm-start predictions

# Core classes
GNNModel                                     # Graph neural network architecture
Graph_custom                                 # Custom PyTorch Geometric dataset
Graph_basic                                  # Basic graph dataset implementation

# Preprocessing functions
process_dili_data()                          # SMILES standardization pipeline
clean_dili_data()                           # Data cleaning and validation
standardise_mol()                           # RDKit molecular standardization
```

### Model Weights & Architecture (from ori)
```
diligenn_models/dlst/
├── initial_model_p20_20/
│   ├── GraphSAGE_Optimised_outer0_seed3.pt    # Base model fold 0
│   ├── GraphSAGE_Optimised_outer1_seed3.pt    # Base model fold 1
│   ├── GraphSAGE_Optimised_outer2_seed7.pt    # Base model fold 2
│   ├── GraphSAGE_Optimised_outer3_seed2.pt    # Base model fold 3
│   └── warm_starts_p50/
│       ├── GraphSAGE_Optimised_outer0_inner[0-4]_run3.pt  # Warm-start models
│       ├── GraphSAGE_Optimised_outer1_inner[0-4]_run3.pt
│       ├── GraphSAGE_Optimised_outer2_inner[0-4]_run3.pt
│       └── GraphSAGE_Optimised_outer3_inner[0-4]_run3.pt
├── grid_search_results.pkl                     # Hyperparameter optimization
└── results_init.pkl                           # Initial training results
```

**Architecture Details**:
- **Base Model**: GraphSAGE_Optimised
- **Training Strategy**: 4-fold outer CV × 5-fold inner CV
- **Warm-start Enhancement**: Transfer learning from base models
- **Supported GNN Types**: GraphSAGE, GCN, GAT, GIN

### Prerequisites
```bash
# Core dependencies
torch>=1.12.0
torch-geometric>=2.0.0
rdkit>=2023.9.2
captum                    # For model interpretability
networkx                  # Graph operations
scikit-learn>=1.2.0
pandas>=1.5.2
numpy>=1.23.5
```

### Installation & Setup
```bash
# Install PyTorch and dependencies
pip install torch torch-geometric rdkit captum

# Verify CUDA availability (recommended)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Running Predictions
```python
from insilico_benchmarks import (
    diligenn_predict_outer_folds_warm_starts, 
    Graph_custom, 
    GNNModel,
    process_dili_data,
    clean_dili_data
)

# 1. Preprocess data
df_processed = process_dili_data(your_dataframe, smiles_col='smiles')
df_clean = clean_dili_data(df_processed, smiles_col='smiles', label_col='label')

# 2. Run warm-start predictions (recommended)
predictions = diligenn_predict_outer_folds_warm_starts(
    df_clean,
    dataset_class=Graph_custom,
    model_name='GraphSAGE_Optimised',
    model_class=GNNModel,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    model_selection_strategy='last_inner_fold'
)
```

---

## 2. DILIPredictor (RF Ensemble)

### Original Work
```
Improved Detection of Drug-Induced Liver Injury by Integrating Predicted In Vivo and In Vitro Data Srijit Seal, Dominic Williams, Layla Hosseini-Gerami, Manas Mahale, Anne E. Carpenter, Ola Spjuth, and Andreas Bender doi: https://doi.org/10.1021/acs.chemrestox.4c00015
```
**GitHub**: https://github.com/Manas02/dili-pip.git  


### Model Description
DILIPredictor uses traditional machine learning approaches with molecular descriptors. It employs feature engineering and ensemble methods for DILI prediction based on chemical properties.

### Reproducibility Scripts
- **Main Script**: `run_dilipredictor.py`
- **Environment Config**: `dilipr_environment.yaml` 

### Key Functions
```python
# Main execution functions
run_dilipredictor_predictions()    # Execute full prediction pipeline
setup_dilipredictor_environment()  # Environment configuration
validate_environment()             # Dependency validation
check_conda_environment()          # Environment verification
```

### Environment Configuration
**File**: `dilipr_environment.yaml`
```yaml
name: dilipr-conda
dependencies:
  - python>=3.10,<3.12
  - scikit-learn==1.2.0      # Specific version for model compatibility
  - rdkit==2023.9.2
  - MolVS==0.1.1             
  - mordred==1.2.0           
  - shap==0.41.0             
  - pandas>=1.5.2
  - numpy==1.23.5
  - loguru>=0.7.2,<0.8.0
  - dimorphite-dl>=1.2.5     
```

### Prerequisites & Installation
```bash
# 1. Clone DILIPredictor repository
git clone https://github.com/Manas02/dili-pip.git
cd dili-pip/

# 2. Install poetry2conda converter
pip install poetry2conda

# 3. Convert poetry to conda environment
poetry2conda pyproject.toml environment.yaml

# 4. Create conda environment
conda env create -f /path/to/dilipr_environment.yaml
conda activate dilipr-conda

# 5. Install compatible scikit-learn (critical for model compatibility)
pip install scikit-learn==1.2.0
```

### Running Predictions
```bash
# Command line interface
python run_dilipredictor.py \
    -i /path/to/input_smiles.csv \
    -o /path/to/output_predictions.csv \
    --dili_pip_repo /path/to/dili-pip \
    --n_jobs 4

# The script automatically:
# 1. Validates conda environment (dilipr-conda)
# 2. Processes SMILES strings
# 3. Generates molecular descriptors
# 4. Runs ensemble predictions
# 5. Outputs results with probability scores
```

---

## 3. TxGemma 2B, 9B, 27B variants (Generalist LLM)


### Original Work
```bibtex
@article{wang2025txgemma,
    title={TxGemma: Efficient and Agentic LLMs for Therapeutics},
    author={Wang, Eric and Schmidgall, Samuel and Jaeger, Paul F. and Zhang, Fan and Pilgrim, Rory and Matias, Yossi and Barral, Joelle and Fleet, David and Azizi, Shekoofeh},
    year={2025},
}
```
**GitHub**: [https://github.com/google-gemini/gemma-cookbook/tree/main/TxGemma]  


### Model Description
TxGemma is a generalist LLM from Google DeepMind fine-tuned from Gemma-2 model on tasks from the Therapeutic Data Commons (TDC) including DILI prediction. It supports multiple model variants (2B, 9B, 27B parameters) deployed on Vertex AI.

### Reproducibility Scripts
- **Main Script**: `run_txgemma_dili.py` 

### Key Classes & Functions
```python
# Main predictor class
class TxGemmaDILIPredictor:
    def __init__(endpoint_id, endpoint_region, project_id)
    def setup_vertex_ai()              # Initialize Vertex AI connection
    def load_prompt_templates()        # Load TDC prompt templates
    def predict_dili_batch()           # Batch prediction processing
    def save_results_to_gcs()          # Save results to Google Cloud Storage

# Pipeline functions
run_dili_benchmark()                   # Complete benchmark pipeline
setup_gcp_environment()                # GCP authentication setup
```

### Prerequisites & Setup
TxGemma models require either Google Cloud Platform / Vertex AI deployment or Hugging Face deployment. All TxGemma variants (2B, 9B, 27B) were deployed on Vertex AI for this benchmark. The following identifiers and credentials are necessary:

- **PROJECT_ID**: GCP project identifier
- **ENDPOINT_ID**: Vertex AI endpoint identifier  
- **ENDPOINT_REGION**: Deployment region (e.g., us-central1)
- **Service Account**: Authentication credentials for API access




