import pandas as pd
import statsmodels.formula.api as smf
from tqdm import tqdm
import os

class PostHocAnalyzer:
    def __init__(self, omics_data_dict):
        """
        omics_data_dict: {'cytokines': df, ...}
        Each df must contain [SubjectID, Time, feature1, feature2, ...].
        """
        self.raw_data = omics_data_dict
        self.p_val_cache = {}

    def _run_lmm_single(self, df, feature, label_df):
        """
        Compute a single linear mixed model:
            Value ~ Group * Time + (1 | Subject)
        """
        # Prepare long-format data
        sub = df[['SubjectID', 'Time', feature]].rename(columns={feature: 'Value'})
        sub = sub.merge(label_df[['SubjectID', 'IRIS']], on='SubjectID')
        
        if len(sub) < 10:
            return 1.0
        
        try:
            model = smf.mixedlm("Value ~ IRIS * Time", sub, groups=sub["SubjectID"])
            res = model.fit(reml=False)
            # Extract P-value of interaction term
            term = [x for x in res.pvalues.index if ":" in x]
            return res.pvalues[term[0]] if term else 1.0
        except:
            return 1.0

    def precompute_lmm(self, label_df, output_dir):
        """
        Precompute LMM p-values for all features and cache results.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in self.raw_data.items():
            cache_file = os.path.join(output_dir, f"{name}_lmm_pvals.csv")
            if os.path.exists(cache_file):
                print(f"Loading cached results: {name}")
                self.p_val_cache[name] = pd.read_csv(cache_file, index_col=0)['p_value'].to_dict()
                continue
            
            print(f"Computing LMM for: {name}")
            pvals = {}
            for feat in tqdm(df.columns[2:]):  # Assume first two columns are ID and Time
                pvals[feat] = self._run_lmm_single(df, feat, label_df)
            
            self.p_val_cache[name] = pvals
            pd.DataFrame.from_dict(pvals, orient='index', columns=['p_value']).to_csv(cache_file)

    def classify_pairs(self, wilcoxon_file, omics1_name, omics2_name, output_file):
        """
        Classify significant feature pairs according to their LMM results.
        """
        df = pd.read_csv(wilcoxon_file)
        sig_df = df[df['significant'] == True].copy()
        
        cats = []
        for _, row in sig_df.iterrows():
            p1 = self.p_val_cache[omics1_name].get(row['feature_g'], 1.0)
            p2 = self.p_val_cache[omics2_name].get(row['feature_h'], 1.0)
            
            sig1 = p1 < 0.05
            sig2 = p2 < 0.05
            
            if not sig1 and not sig2:
                cats.append("Subtle_Coordinated")
            elif sig1 and sig2:
                cats.append("Both_Driven")
            else:
                cats.append("Unilateral_Driver")
        
        sig_df['Category'] = cats
        sig_df.to_csv(output_file, index=False)
        return sig_df['Category'].value_counts()