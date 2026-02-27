# src/test.py
from data_loader import load_independent_test_data
from preprocessing import build_preprocessor
import joblib

# Load test data
X_test, y_test = load_independent_test_data('data_set_ALL_AML_independent.csv', 'actual.csv')

# Load saved preprocessing and target encoder
preprocessor = joblib.load('preprocessor.joblib')
target_encoder = joblib.load('target_encoder.joblib')

X_test_processed = preprocessor.transform(X_test)
y_test_encoded = target_encoder.transform(y_test)

# Load stacking model (final ensemble)
stacking_model = joblib.load('stacking_model.joblib')

# Evaluate
from sklearn.metrics import classification_report
preds = stacking_model.predict(X_test_processed)
print('Stacking Ensemble on Independent Test Classification Report:\n',
      classification_report(y_test_encoded, preds, target_names=target_encoder.classes_))
