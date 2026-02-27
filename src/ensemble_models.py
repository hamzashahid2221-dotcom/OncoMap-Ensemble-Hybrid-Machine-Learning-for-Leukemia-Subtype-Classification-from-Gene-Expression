from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression


def build_voting_model(model_svc, model_cat, model_nn):

    voting_clf = VotingClassifier(
        estimators=[
            ('svc', model_svc),
            ('cat', model_cat),
            ('nn', model_nn)
        ],
        voting='soft'
    )

    return voting_clf


def build_stacking_model(model_svc, model_cat, model_nn):

    stacking_clf = StackingClassifier(
        estimators=[
            ('svc', model_svc),
            ('cat', model_cat),
            ('nn', model_nn)
        ],
        final_estimator=LogisticRegression(),
        passthrough=False
    )

    return stacking_clf
