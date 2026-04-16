from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def get_models():
    return {
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', C=10, probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }