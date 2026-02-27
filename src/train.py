# src/train.py
from data_loader import load_training_data
from preprocessing import build_preprocessor, encode_target, split_data
from models.svm_model import build_svm, train_svm
from models.catboost_model import build_catboost, train_catboost
from models.deep_learning_model import build_keras_classifier
from ensemble.voting import build_voting_ensemble
from ensemble.stacking import build_stacking_ensemble
import joblib

# Load data
X, y = load_training_data('data_set_ALL_AML_train.csv', 'actual.csv')

# Preprocess
X_train, X_val, y_train, y_val = split_data(X, y)
preprocessor = build_preprocessor(X)
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
y_train_encoded, target_encoder = encode_target(y_train)
y_val_encoded = target_encoder.transform(y_val)

# Train base models
svm_model = train_svm(build_svm(), X_train_processed, y_train_encoded)
cat_model = train_catboost(build_catboost(), X_train_processed, y_train_encoded, X_val_processed, y_val_encoded)
keras_clf = build_keras_classifier(X_train_processed.shape[1])
keras_clf.fit(X_train_processed, y_train_encoded)

# Save preprocessor and encoders
joblib.dump(preprocessor, 'preprocessor.joblib')
joblib.dump(target_encoder, 'target_encoder.joblib')

# Ensemble models
models_dict = {'svc': svm_model, 'cat': cat_model, 'dl': keras_clf}

voting_model = build_voting_ensemble(models_dict)
voting_model.fit(X_train_processed, y_train_encoded)

stacking_model = build_stacking_ensemble(models_dict)
stacking_model.fit(X_train_processed, y_train_encoded)

# Save ensemble models
joblib.dump(voting_model, 'voting_model.joblib')
joblib.dump(stacking_model, 'stacking_model.joblib')
