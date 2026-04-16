import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

def evaluate(models, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc):
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        if name in ['SVM (RBF)', 'Logistic Regression']:
            model.fit(X_train_sc, y_train)
            test_pred = model.predict(X_test_sc)
            prob = model.predict_proba(X_test_sc)[:,1]
            cv_scores = cross_val_score(model, X_train_sc, y_train, cv=cv)
        else:
            model.fit(X_train, y_train)
            test_pred = model.predict(X_test)
            prob = model.predict_proba(X_test)[:,1]
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv)

        results.append({
            "Model": name,
            "Train Accuracy": round(model.score(X_train, y_train), 4),
            "Test Accuracy": round(accuracy_score(y_test, test_pred), 4),
            "CV Mean": round(cv_scores.mean(), 4),
            "AUC": round(roc_auc_score(y_test, prob), 4)
        })

    return pd.DataFrame(results)