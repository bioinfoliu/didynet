import pandas as pd
import numpy as np
import os

def compute_variances(df, id_col='SubjectID', time_col='Time'):
    """Compute variance across time and variance across subjects."""
    subjects = df[id_col].unique()
    timepoints = df[time_col].unique()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in (id_col, time_col)]
    
    rows = []
    for feat in feature_cols:
        # 1. Variance across time (Var_Time)
        vt_list = []
        for t in timepoints:
            vals = df.loc[df[time_col] == t, feat].values
            if len(vals) > 1:
                vt_list.append(np.var(vals, ddof=1))
        var_time_mean = np.nanmean(vt_list) if vt_list else np.nan

        # 2. Variance across subjects (Var_Subject)
        vi_list = []
        for s in subjects:
            vals = df.loc[df[id_col] == s, feat].values
            if len(vals) > 1:
                vi_list.append(np.var(vals, ddof=1))
        var_subject_mean = np.nanmean(vi_list) if vi_list else np.nan
        
        rows.append({
            'Feature': feat,
            'Var_Time_mean': var_time_mean,
            'Var_Subject_mean': var_subject_mean
        })
    return pd.DataFrame(rows).dropna()

def get_top_features(var_df, k, mode='union'):
    """Get the top K features based on variance ranking."""
    k = min(int(k), len(var_df))
    time_top = set(var_df.nlargest(k, 'Var_Time_mean')['Feature'])
    subj_top = set(var_df.nlargest(k, 'Var_Subject_mean')['Feature'])
    
    if mode == 'union':
        return sorted(list(time_top.union(subj_top)))
    elif mode == 'intersection':
        return sorted(list(time_top.intersection(subj_top)))
    return []