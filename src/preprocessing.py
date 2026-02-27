# src/preprocessing.py
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def build_preprocessor(X):
    num_cols = X.select_dtypes(include=['int64','float64']).columns
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), num_cols)])
    return preprocessor

def encode_target(y):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder

def split_data(X, y, test_size=0.2, random_state=1):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
