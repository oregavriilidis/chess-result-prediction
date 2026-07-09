"""
train_model.py
--------------
Εκπαιδεύει ένα μοντέλο ταξινόμησης (Random Forest) που προβλέπει το
αποτέλεσμα μιας παρτίδας (white / black / draw) βάσει:
  - rating_diff (διαφορά rating λευκού - μαύρου)
  - avg_rating
  - opening_name
  - time_control
  - num_moves

Αποθηκεύει το εκπαιδευμένο μοντέλο με joblib και τυπώνει metrics.
"""

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = ["rating_diff", "avg_rating", "num_moves"]
CATEGORICAL_FEATURES = ["opening_name", "time_control"]
TARGET = "result"


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=42, class_weight="balanced"
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def main():
    df = pd.read_csv("data/clean_games.csv")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy στο test set: {acc:.3f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Confusion Matrix - Πρόβλεψη Αποτελέσματος")
    plt.tight_layout()
    plt.savefig("images/confusion_matrix.png", dpi=150)
    plt.close()

    # Feature importance (μόνο αριθμητικά + top categorical, για ευανάγνωστο γράφημα)
    ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERIC_FEATURES + cat_names
    importances = pipeline.named_steps["model"].feature_importances_

    imp_df = pd.DataFrame({"feature": all_feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(12)

    plt.figure(figsize=(7, 5))
    plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#4C72B0")
    plt.title("Top 12 Σημαντικότερα Χαρακτηριστικά (Feature Importance)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("images/feature_importance.png", dpi=150)
    plt.close()

    joblib.dump(pipeline, "src/chess_result_model.joblib")
    print("\nΤο μοντέλο αποθηκεύτηκε -> src/chess_result_model.joblib")
    print("Confusion matrix & feature importance -> images/")


if __name__ == "__main__":
    main()
