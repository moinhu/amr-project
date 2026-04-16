from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.preprocessing import preprocess
from src.models import get_models
from src.evaluation import evaluate

app = Flask(__name__)

# Ensure static folder exists
if not os.path.exists("static"):
    os.makedirs("static")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    if not file:
        return "No file uploaded"

    # Read CSV
    df = pd.read_csv(file)

    # Safety check
    if 'label' not in df.columns:
        return "❌ ERROR: Dataset must contain 'label' column"

    # ==============================
    # 🔁 ML PIPELINE
    # ==============================
    X_train, X_test, y_train, y_test, X_train_sc, X_test_sc = preprocess(df)
    models = get_models()
    results_df = evaluate(models, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc)

    # Best model
    best_model = results_df.loc[results_df['Test Accuracy'].idxmax()]
    best_model_name = best_model['Model']
    model_obj = models[best_model_name]

    # ==============================
    # 📊 CHART 1: Accuracy
    # ==============================
    plt.figure()
    plt.bar(results_df['Model'], results_df['Test Accuracy'])
    plt.xticks(rotation=20)
    plt.title("Model Accuracy Comparison")
    plt.tight_layout()
    plt.savefig("static/accuracy.png")
    plt.close()

    # ==============================
    # 📊 CHART 2: AUC
    # ==============================
    plt.figure()
    plt.bar(results_df['Model'], results_df['AUC'])
    plt.xticks(rotation=20)
    plt.title("AUC Comparison")
    plt.tight_layout()
    plt.savefig("static/auc.png")
    plt.close()

    # ==============================
    # 📉 CONFUSION MATRIX
    # ==============================
    if best_model_name in ['SVM (RBF)', 'Logistic Regression']:
        y_pred = model_obj.predict(X_test_sc)
    else:
        y_pred = model_obj.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("static/confusion.png")
    plt.close()

    # ==============================
    # 📋 TABLE
    # ==============================
    table_html = results_df.to_html(classes='table', index=False)

    return render_template(
        'result.html',
        table=table_html,
        best_model=best_model_name,
        accuracy=round(best_model['Test Accuracy'] * 100, 2)
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)