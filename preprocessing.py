import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, label_df: pd.DataFrame):
        """
        label_df: 包含 ['SubjectID', 'Time', 'IRIS'] 列的 dataframe
        """
        self.label_df = label_df
        # 筛选出只有 IS 和 IR 的数据
        self.ir_data = label_df[label_df['IRIS'].isin(['IS', 'IR'])]

    def process(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        print(f"[{name}] 开始预处理...")
        
        # 1. 筛选有标签样本
        filtered = df.merge(self.ir_data[['SubjectID', 'Time']], on=['SubjectID', 'Time'])
        
        # 2. 去重 (按 SubjectID, Time 取均值)
        key_cols = ['SubjectID', 'Time']
        value_cols = filtered.select_dtypes(include='number').columns.difference(key_cols)
        deduped = filtered.groupby(key_cols)[value_cols].mean().reset_index()
        
        # 3. 标准化 (Z-score)
        # 假设前两列是 ID 和 Time，后面是特征
        scaler = StandardScaler()
        feature_data = deduped.iloc[:, 2:]
        deduped.iloc[:, 2:] = scaler.fit_transform(feature_data)
        
        # 4. 列名清洗 (去除特殊字符)
        deduped.columns = deduped.columns.str.replace(r'[^a-zA-Z0-9]', '_', regex=True)
        
        print(f"[{name}] 处理完成. Shape: {deduped.shape}")
        return deduped