# DILImap Reproducibility Notebooks

This repository contains Jupyter notebooks supporting the reproducibility of all results in the DILImap project. 
- DILImap codebase: [github.com/Cellarity/DILImap](https://github.com/Cellarity/DILImap)
- DILImap website: [dilimap.org](https://dilimap.org/review-dUFZulWv8k7bERJ3FQs4)

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