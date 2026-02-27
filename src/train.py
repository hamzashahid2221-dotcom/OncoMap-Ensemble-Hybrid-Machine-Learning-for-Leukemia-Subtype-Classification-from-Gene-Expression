from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

from src.models import build_svm, build_catboost
from src.ensemble_models import build_voting_model, build_stacking_model

# Assume preprocessing already done

# Train base models
model_svc = build_svm()
model_cat = build_catboost()

model_svc.fit(train_X_processed, train_y_encoded)
model_cat.fit(train_X_processed, train_y_encoded)

# keras_clf already defined and fitted

# Build ensembles
voting_clf = build_voting_model(model_svc, model_cat, keras_clf)
stacking_clf = build_stacking_model(model_svc, model_cat, keras_clf)

voting_clf.fit(train_X_processed, train_y_encoded)
stacking_clf.fit(train_X_processed, train_y_encoded)

voting_pred = voting_clf.predict(val_X_processed)
stacking_pred = stacking_clf.predict(val_X_processed)

print("Voting Classification Report\n",
      classification_report(val_y_encoded, voting_pred))

print("Stacking Classification Report\n",
      classification_report(val_y_encoded, stacking_pred))

vote_scores = cross_val_score(
    voting_clf,
    train_X_processed,
    train_y_encoded,
    cv=5,
    scoring="accuracy"
)

stack_scores = cross_val_score(
    stacking_clf,
    train_X_processed,
    train_y_encoded,
    cv=5,
    scoring="accuracy"
)

print("Average Voting Score:", vote_scores.mean())
print("Average Stacking Score:", stack_scores.mean())
