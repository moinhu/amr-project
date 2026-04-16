from src.data_generation import create_dataset
from src.preprocessing import preprocess
from src.models import get_models
from src.evaluation import evaluate

def main():
    df = create_dataset()

    X_train, X_test, y_train, y_test, X_train_sc, X_test_sc = preprocess(df)

    models = get_models()

    results_df = evaluate(models, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc)

    print("\n🔥 FINAL RESULTS TABLE:\n")
    print(results_df)

    results_df.to_csv("results_table.csv", index=False)

if __name__ == "__main__":
    main()