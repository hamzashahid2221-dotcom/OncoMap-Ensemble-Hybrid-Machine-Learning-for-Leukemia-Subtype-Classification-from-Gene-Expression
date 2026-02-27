# src/ensemble/voting.py
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

def build_voting_ensemble(models):
    """
    models: dict of name:model, e.g. {'svc': model_svc, 'cat': model_cat, 'dl': keras_clf}
    """
    voting_clf = VotingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        voting='soft'
    )
    return voting_clf

def evaluate_voting(model, X_val, y_val, target_names):
    preds = model.predict(X_val)
    print('Voting Ensemble Classification Report:\n', classification_report(y_val, preds, target_names=target_names))
    scores = cross_val_score(model, X_val, y_val, cv=5, scoring='accuracy')
    print('Average Voting CV Score:', scores.mean())
