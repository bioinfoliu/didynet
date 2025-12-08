# run_pipeline.py
import pandas as pd
import os
from didynet import DataPreprocessor, compute_variances, get_top_features, run_dtw_analysis, run_wilcoxon_testing, PostHocAnalyzer

# 1. Set paths
BASE_DIR = "./data"
OUT_DIR = "./output"
label_df = pd.read_csv(os.path.join(BASE_DIR, "IRIS_label.csv"))

# 2. Preprocessing
processor = DataPreprocessor(label_df)
omics_data = {}
for name in ["cytokines", "proteomics", "transcriptome"]:
    raw = pd.read_csv(os.path.join(BASE_DIR, f"{name}.csv"))
    omics_data[name] = processor.process(raw, name)

# 3. Feature selection
feature_lists = {name: {} for name in omics_data}
for name, df in omics_data.items():
    var_df = compute_variances(df)
    for k in [50, 100, 200]:
        feature_lists[name][k] = get_top_features(var_df, k, mode='union')

# 4. DTW computation (the most time-consuming step)
run_dtw_analysis(
    omics_data,
    feature_lists,
    output_dir=os.path.join(OUT_DIR, "dtw"),
    ks=[100],
    mode='union'
)

# 5. Statistical testing
run_wilcoxon_testing(
    dtw_dir=os.path.join(OUT_DIR, "dtw"),
    label_df=label_df,
    output_dir=os.path.join(OUT_DIR, "wilcoxon")
)

# 6. Post-hoc analysis
analyzer = PostHocAnalyzer(omics_data)
analyzer.precompute_lmm(label_df, output_dir=os.path.join(OUT_DIR, "lmm"))

# Classify pairs for a specific Wilcoxon result file
analyzer.classify_pairs(
    wilcoxon_file=os.path.join(OUT_DIR, "wilcoxon/cytokines_proteomics_k100_union_wilcoxon.csv"),
    omics1_name="cytokines",
    omics2_name="proteomics",
    output_file=os.path.join(OUT_DIR, "final_network.csv")
)

# 7. Visualization (Network Analysis)
from didynet import NetworkPlotter

print("\n=== Drawing network graph ===")

# Initialize plotter (passing omics_data to automatically identify feature group)
plotter = NetworkPlotter(omics_data)

# Suppose the PostHoc step generated final_network_edges.csv
plotter.build_network_from_file(
    edge_file=os.path.join(OUTPUT_DIR, "final_network_edges.csv"),
    target_category="Subtle_Coordinated"  # Plot only the most important Subtle category
)

# Plot and save
plotter.plot_top_hubs(
    output_file=os.path.join(OUTPUT_DIR, "figures/Figure4A_Network.png"),
    top_k=20  # Select top 20 hubs from each layer
)

print("Pipeline Finished!")