from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def get_models():
    return {
        'Random Forest': RandomForestClassifier(
            n_estimators=120,
            max_depth=6,                 # prevents overfitting
            min_samples_split=5,
            random_state=42
        ),

        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=2,                 # keeps model simple
            random_state=42
        ),

        'SVM (RBF)': SVC(
            kernel='rbf',
            C=1,                         # reduced to avoid overfitting
            gamma='scale',
            probability=True,
            random_state=42
        ),

        'Logistic Regression': LogisticRegression(
            C=0.8,                       # adds regularization
            max_iter=1000,
            random_state=42
        )
    }