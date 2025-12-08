# DiDyNet: A Robust Framework for Differential Dynamic Network Inference from Longitudinal Multi-omics Data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**DiDyNet** is a robust computational framework designed to identify time-evolving molecular networks that differ between phenotypic groups (e.g., Disease vs. Control). It is specifically tailored for **longitudinal multi-omics data**, overcoming challenges such as asynchronous biological signals, data sparsity, and high-dimensional noise.

## 🚀 Key Features

* **Asynchronous Alignment:** Uses **Dynamic Time Warping (DTW)** to capture non-linear and time-lagged associations between molecules.
* **Noise Resilience:** Implements a rigorous **Linear Mixed Model (LMM)** post-hoc refinement step to distinguish genuine biological coordination from artifacts driven by stochastic fluctuations.
* **Multi-omics Support:** Designed to handle complex interactions within and between omics layers (e.g., Transcriptome-Proteomics).
* **Scalable:** Includes a 2D variance-based filtering strategy to efficiently handle high-dimensional data.

---

## 📦 Installation

You can install DiDyNet directly from source.

### Prerequisites
* Python >= 3.8
* We recommend using a virtual environment (Conda or venv).

### Install via pip

```bash
# Clone the repository
git clone [https://github.com/](https://github.com/)bioinfoliu/DiDyNet.git
cd DiDyNet

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .



### Data Preparation
# Input data should be in CSV format (Long format or Wide format with metadata).

# Required Columns:
# SubjectID: Unique identifier for each individual.
# Time: Numeric time points (e.g., days, weeks).
# Feature Columns: Gene expression, protein abundance, # cytokine levels, etc.
# Label Data: A separate CSV file mapping subjects to groups is required:
# Columns: SubjectID, IRIS (or Group), etc.


### Usage Example
### Here is a minimal example to run the full DiDyNet pipeline:


import pandas as pd
import os
from didynet import (
    DataPreprocessor, 
    compute_variances, 
    get_top_features, 
    run_dtw_analysis, 
    run_wilcoxon_testing, 
    PostHocAnalyzer
)

# 1. Setup Paths
BASE_DIR = "./data"
OUTPUT_DIR = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Load Label Data
label_df = pd.read_csv(os.path.join(BASE_DIR, "IRIS_label.csv"))

# 3. Preprocessing
processor = DataPreprocessor(label_df)
omics_data = {}
for name in ["cytokines", "proteomics"]:
    # Assuming raw files exist
    raw_df = pd.read_csv(os.path.join(BASE_DIR, f"{name}.csv"))
    omics_data[name] = processor.process(raw_df, name)

# 4. Feature Selection
feature_lists = {name: {} for name in omics_data}
for name, df in omics_data.items():
    var_df = compute_variances(df)
    feature_lists[name][100] = get_top_features(var_df, k=100, mode='union')

# 5. Dynamic Time Warping Analysis
run_dtw_analysis(
    omics_data, 
    feature_lists, 
    output_dir=os.path.join(OUTPUT_DIR, "dtw"), 
    ks=[100], 
    mode='union',
    n_jobs=-1
)

# 6. Statistical Testing & Post-hoc Refinement
run_wilcoxon_testing(
    dtw_dir=os.path.join(OUTPUT_DIR, "dtw"),
    label_df=label_df,
    output_dir=os.path.join(OUTPUT_DIR, "wilcoxon")
)

analyzer = PostHocAnalyzer(omics_data)
analyzer.precompute_lmm(label_df, output_dir=os.path.join(OUTPUT_DIR, "lmm_cache"))

analyzer.classify_pairs(
    wilcoxon_file=os.path.join(OUTPUT_DIR, "wilcoxon/cytokines_proteomics_k100_union_wilcoxon.csv"),
    omics1_name="cytokines",
    omics2_name="proteomics",
    output_file=os.path.join(OUTPUT_DIR, "final_network.csv")
)

print("Pipeline Finished!") 

```


### Citation
If you use DiDyNet in your research, please cite our paper: DiDyNet: A Robust Framework for Differential Dynamic Network Inference from Longitudinal Multi-omics Data Zhe Liu, Kesong Wu, Taesung Park. , 2025. 


---

### Contact
For any questions or issues, please open an issue on GitHub or contact the corresponding author: Taesung Park (tspark@stats.snu.ac.kr)

