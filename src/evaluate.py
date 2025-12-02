from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test):
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)

    print("\nROC-AUC Score:", roc_auc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return roc_auc