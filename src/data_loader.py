# src/data_loader.py
import pandas as pd

def load_training_data(expression_file: str, label_file: str):
    """Load and merge training dataset."""
    df = pd.read_csv(expression_file)
    df_clean = df.drop(columns=[col for col in df.columns if 'call' in col]+['Gene Description'])
    df_trans = df_clean.set_index('Gene Accession Number').T.reset_index().rename(columns={'index':'patient'})
    df_trans['patient'] = df_trans['patient'].astype(int)

    labels = pd.read_csv(label_file)
    labels['patient'] = labels['patient'].astype(int)

    merged = df_trans.merge(labels, on='patient')
    X = merged.drop('cancer', axis=1)
    y = merged['cancer']
    return X, y

def load_independent_test_data(expression_file: str, label_file: str):
    """Load independent test dataset."""
    return load_training_data(expression_file, label_file)
