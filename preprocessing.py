import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, label_df: pd.DataFrame):
        """
        Initialize the preprocessor.
        label_df: A pandas DataFrame containing ['SubjectID', 'Time', 'IRIS'] columns.
        """
        self.label_df = label_df
        # Filter the dataset to include only 'IS' (Insulin Sensitive) and 'IR' (Insulin Resistant) subjects
        self.ir_data = label_df[label_df['IRIS'].isin(['IS', 'IR'])]

    def process(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        print(f"[{name}] Starting preprocessing...")
        
        # 1. Filter samples by merging with the valid labeled dataset
        filtered = df.merge(self.ir_data[['SubjectID', 'Time']], on=['SubjectID', 'Time'])
        
        # 2. Deduplication (Calculate the mean for duplicate measurements at the same time point for the same subject)
        key_cols = ['SubjectID', 'Time']
        value_cols = filtered.select_dtypes(include='number').columns.difference(key_cols)
        deduped = filtered.groupby(key_cols)[value_cols].mean().reset_index()
        
        # 3. Z-score Normalization (Standardization)
        # Assuming the first two columns are ID and Time, and the remaining are molecular features
        scaler = StandardScaler()
        feature_data = deduped.iloc[:, 2:]
        deduped.iloc[:, 2:] = scaler.fit_transform(feature_data)
        
        # 4. Clean column names (Replace any special characters with underscores to ensure compatibility)
        deduped.columns = deduped.columns.str.replace(r'[^a-zA-Z0-9]', '_', regex=True)
        
        print(f"[{name}] Preprocessing completed. Final Shape: {deduped.shape}")
        return deduped
