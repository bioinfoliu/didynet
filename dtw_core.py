import pandas as pd
import numpy as np
from dtaidistance import dtw
from joblib import Parallel, delayed
from tqdm import tqdm
import os

def _compute_pair_dtw(df1, df2, subj, f1, f2, id_col, time_col):
    """Helper function for computing DTW for a single subject."""
    try:
        s1 = df1[df1[id_col] == subj].sort_values(time_col)[f1].dropna().values
        s2 = df2[df2[id_col] == subj].sort_values(time_col)[f2].dropna().values
        
        if len(s1) < 2 or len(s2) < 2:
            return None
        
        dist = dtw.distance(s1, s2)
        return (f1, f2, subj, dist)
    except:
        return None

def run_dtw_analysis(omics_data_dict, feature_lists, output_dir, ks, mode='union', n_jobs=-1):
    """
    Main DTW execution function.
    
    omics_data_dict: {'cytokines': df, ...}
    feature_lists: {'cytokines': {50: [feat_list], ...}, ...}
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get intersection of all subjects across data types
    sets = [set(df['SubjectID']) for df in omics_data_dict.values()]
    common_subjects = list(set.intersection(*sets))
    
    omics_names = list(omics_data_dict.keys())
    pairs = [(omics_names[i], omics_names[j]) for i in range(len(omics_names)) for j in range(i, len(omics_names))]
    
    for k in ks:
        print(f"\n🚀 Running DTW for K={k} ({mode})...")
        
        for name1, name2 in pairs:
            # Get feature lists
            feats1 = feature_lists[name1][k]
            feats2 = feature_lists[name2][k]
            
            # Build feature pair combinations
            feat_pairs = [(f1, f2) for f1 in feats1 for f2 in feats2]
            
            print(f"  -> Processing {name1} vs {name2} ({len(feat_pairs)} pairs)...")
            
            # Parallel DTW computation
            results = Parallel(n_jobs=n_jobs)(
                delayed(_compute_pair_dtw)(
                    omics_data_dict[name1],
                    omics_data_dict[name2],
                    s, f1, f2, 'SubjectID', 'Time'
                ) for f1, f2 in tqdm(feat_pairs) for s in common_subjects
            )
            
            # Save results
            results = [r for r in results if r is not None]
            res_df = pd.DataFrame(results, columns=['feature1', 'feature2', 'subject', 'distance'])
            
            filename = f"{name1}_{name2}_k{k}_{mode}_dtw.csv"
            res_df.to_csv(os.path.join(output_dir, filename), index=False)