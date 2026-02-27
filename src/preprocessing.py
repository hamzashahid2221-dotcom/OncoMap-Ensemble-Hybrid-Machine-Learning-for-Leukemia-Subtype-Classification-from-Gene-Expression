from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def preprocess_features(X):
    num_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    num_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", num_transformer, num_cols)]
    )

    return preprocessor


def encode_target(y):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder
