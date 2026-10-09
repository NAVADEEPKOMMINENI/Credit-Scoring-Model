from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_dataset():
    """Load the local CSV when it contains a target, otherwise fetch the official UCI German Credit dataset."""
    csv_path = Path(__file__).resolve().parent / "credit" / "german_credit_data (1).csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        if "Risk" in df.columns:
            return df

    print("Local dataset does not contain a target column. Fetching the official UCI German Credit dataset...")
    data = fetch_ucirepo(id=144)
    X = data.data.features.copy()
    feature_names = data.variables[data.variables["name"] != "class"]["name"].tolist()
    X.columns = feature_names
    y = data.data.targets["class"].copy()
    y = (y == 1).astype(int)
    df = pd.concat([X, y.rename("Risk")], axis=1)
    return df


def clean_target(df):
    if "Risk" not in df.columns:
        raise ValueError("Dataset must contain a 'Risk' target column.")

    y = df["Risk"].copy()
    if y.dtype == object:
        mapping = {"good": 1, "bad": 0, "Good": 1, "Bad": 0, "1": 1, "0": 0}
        y = y.map(mapping)
    else:
        y = y.astype(int)

    if set(y.unique()) <= {0, 1}:
        return y

    # If the target is encoded as 1/2, convert 1=Good and 2=Bad.
    y = y.replace({1: 1, 2: 0})
    return y


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    transformers = []
    if numeric_features:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features))
    if categorical_features:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def evaluate_model(name, y_true, y_pred, y_prob):
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{title.lower().replace(' ', '_')}.png", dpi=200)
    plt.close()


def plot_roc_curve(models, y_true):
    plt.figure(figsize=(8, 6))
    for label, probs in models.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=200)
    plt.close()


def plot_feature_importance(model, preprocessor, top_n=15):
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    importance_df = importance_df.sort_values("Importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 7))
    sns.barplot(data=importance_df, x="Importance", y="Feature", hue="Feature", dodge=False, palette="viridis", legend=False)
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=200)
    plt.close()

    return importance_df


def main():
    sns.set_theme(style="whitegrid")

    df = load_dataset()
    y = clean_target(df)
    X = df.drop(columns=["Risk"]).copy()

    if not X.columns.size:
        raise ValueError("No features were found. The dataset is empty or malformed.")

    print("Dataset shape:", df.shape)
    print("Target distribution:\n", y.value_counts(normalize=True).round(3).to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", DecisionTreeClassifier(max_depth=5, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight="balanced",
                random_state=42,
            ))
        ]),
    }

    model_results = []
    roc_probabilities = {}

    for model_name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        roc_probabilities[model_name] = y_prob
        metrics = evaluate_model(model_name, y_test, y_pred, y_prob)
        model_results.append(metrics)

        print(f"\n{model_name}")
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall: {metrics['Recall']:.4f}")
        print(f"F1 Score: {metrics['F1 Score']:.4f}")
        print(f"ROC-AUC: {metrics['ROC-AUC']:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Bad", "Good"]))

    results_df = pd.DataFrame(model_results)
    print("\nModel Comparison Table:")
    print(results_df.to_string(index=False))
    results_df.to_csv(OUTPUT_DIR / "model_results.csv", index=False)

    best_model_name = results_df.sort_values("ROC-AUC", ascending=False).iloc[0]["Model"]
    best_pipeline = models[best_model_name]
    best_pipeline.fit(X, y)

    plot_confusion_matrix(y_test, models["Random Forest"].predict(X_test), "Random Forest Confusion Matrix")
    plot_roc_curve({k: v for k, v in roc_probabilities.items()}, y_test)

    final_preprocessor = best_pipeline.named_steps["preprocessor"]
    final_model = best_pipeline.named_steps["classifier"]
    feature_importance_df = plot_feature_importance(final_model, final_preprocessor)
    print("\nTop 15 feature importances:")
    print(feature_importance_df.to_string(index=False))

    new_customer = X.iloc[[0]].copy()
    prediction = best_pipeline.predict(new_customer)[0]
    probability = best_pipeline.predict_proba(new_customer)[0]

    print("\nPrediction Example for a New Applicant:")
    print(f"Prediction label: {'Good' if prediction == 1 else 'Bad'}")
    print(f"Probability of Bad Risk: {probability[0]:.4f}")
    print(f"Probability of Good Risk: {probability[1]:.4f}")

    if best_model_name == "Random Forest":
        print("\nSelected Best Model: Random Forest")
    else:
        print(f"\nSelected Best Model: {best_model_name}")

    print(f"\nAll outputs were saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
