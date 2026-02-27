# src/ensemble/stacking.py
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

def build_stacking_ensemble(models, final_estimator=None):
    if final_estimator is None:
        final_estimator = LogisticRegression()
    stacking_clf = StackingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        final_estimator=final_estimator,
        passthrough=False
    )
    return stacking_clf

def evaluate_stacking(model, X_val, y_val, target_names):
    preds = model.predict(X_val)
    print('Stacking Ensemble Classification Report:\n', classification_report(y_val, preds, target_names=target_names))
    scores = cross_val_score(model, X_val, y_val, cv=5, scoring='accuracy')
    print('Average Stacking CV Score:', scores.mean())
