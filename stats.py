import pandas as pd
import numpy as np
from scipy.stats import ranksums
import os
import glob
from tqdm import tqdm

def run_wilcoxon_testing(dtw_dir, label_df, output_dir):
    """读取 DTW 结果并进行 Wilcoxon 检验"""
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(dtw_dir, "*_dtw.csv"))
    
    label_map = label_df.set_index('SubjectID')['IRIS'].to_dict()
    
    for f in files:
        df = pd.read_csv(f)
        df['group'] = df['subject'].map(label_map)
        df = df.dropna(subset=['group'])
        
        stats = []
        # 按特征对分组计算
        for (f1, f2), sub in tqdm(df.groupby(['feature1', 'feature2']), desc=os.path.basename(f)):
            g_is = sub[sub['group'] == 'IS']['distance']
            g_ir = sub[sub['group'] == 'IR']['distance']
            
            if len(g_is) >= 3 and len(g_ir) >= 3:
                s, p = ranksums(g_is, g_ir)
                stats.append({
                    'feature_g': f1, 'feature_h': f2, 
                    'p_value': p, 
                    'mean_diff': g_is.mean() - g_ir.mean()
                })
        
        res_df = pd.DataFrame(stats)
        if not res_df.empty:
            res_df['adj_p'] = np.minimum(res_df['p_value'] * len(res_df), 1.0)
            res_df['significant'] = res_df['adj_p'] < 0.05
            
            out_name = os.path.basename(f).replace("_dtw.csv", "_wilcoxon.csv")
            res_df.to_csv(os.path.join(output_dir, out_name), index=False)