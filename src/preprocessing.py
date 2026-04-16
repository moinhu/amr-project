from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(df):
    feature_cols = [c for c in df.columns if c != 'label']
    X = df[feature_cols]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_sc, X_test_sc