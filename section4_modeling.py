from pathlib import Path
import random

import matplotlib
# Use a non-interactive backend to avoid Tk-related hangs in headless/debug environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted
from scipy import sparse
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from section3_preprocessing import add_engineered_features, build_preprocessor, load_and_clean_data


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """A lightweight sklearn-compatible binary classifier implemented with PyTorch."""

    _estimator_type = "classifier"

    def __init__(
        self,
        hidden_layer_sizes=(64, 32),
        learning_rate=0.001,
        batch_size=128,
        max_epochs=40,
        weight_decay=0.0,
        random_state=42,
        device="cpu",
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device

    @staticmethod
    def _to_float_array(X):
        if sparse.issparse(X):
            X = X.toarray()
        return np.asarray(X, dtype=np.float32)

    def _build_network(self, input_dim: int) -> nn.Module:
        layers = []
        prev_dim = input_dim

        for hidden_dim in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        X_np = self._to_float_array(X)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        self.device_ = torch.device(self.device)
        self.model_ = self._build_network(X_np.shape[1]).to(self.device_)

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        dataset = TensorDataset(
            torch.from_numpy(X_np),
            torch.from_numpy(y_np),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for _ in range(self.max_epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device_)
                batch_y = batch_y.to(self.device_)

                optimizer.zero_grad()
                logits = self.model_(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        self.classes_ = np.array([0, 1], dtype=int)
        self.n_features_in_ = X_np.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["model_", "device_"])
        X_np = self._to_float_array(X)
        tensor_x = torch.from_numpy(X_np).to(self.device_)

        self.model_.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self.model_(tensor_x)).cpu().numpy().reshape(-1)

        return np.column_stack([1.0 - probs, probs])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


sns.set_theme(style="whitegrid")


def prepare_data(base_dir: Path):
    """Load data and apply the same cleaning and feature engineering as Section 3."""
    csv_path = base_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = load_and_clean_data(csv_path)
    df = add_engineered_features(df)

    y = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    X = df.drop(columns=["Churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def evaluate_pipeline(name: str, pipeline: Pipeline, X_train, y_train, X_test, y_test) -> dict:
    """Train and evaluate one model pipeline, returning core metrics."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = pipeline.named_steps.get("model")
    cv_n_jobs = 1 if isinstance(model, TorchMLPClassifier) else -1
    cv_auc_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=cv_n_jobs,
    )

    result = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "cv_roc_auc_mean": float(np.mean(cv_auc_scores)),
        "cv_roc_auc_std": float(np.std(cv_auc_scores)),
        "y_pred": y_pred,
        "y_prob": y_prob,
        "fitted_pipeline": pipeline,
    }
    return result


def tune_advanced_models(preprocessor: ColumnTransformer, X_train, y_train):
    """Run grid-search hyperparameter tuning for advanced models."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    advanced_models = {
        "DecisionTree": (
            DecisionTreeClassifier(random_state=42, class_weight="balanced"),
            {
                "model__max_depth": [4, 6, 8, None],
                "model__min_samples_split": [2, 10, 20],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1),
            {
                "model__n_estimators": [150, 300],
                "model__max_depth": [6, 10, None],
                "model__min_samples_leaf": [1, 3, 5],
            },
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
        "NeuralNetwork": (
            TorchMLPClassifier(
                random_state=42,
                max_epochs=40,
                device="cpu",
            ),
            {
                "model__hidden_layer_sizes": [(32,), (64,), (64, 32)],
                "model__learning_rate": [0.001],
                "model__weight_decay": [0.0, 0.0001],
                "model__batch_size": [64, 128],
            },
        ),
    }

    tuned_results = {}
    for model_name, (model, param_grid) in advanced_models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=1 if model_name == "NeuralNetwork" else -1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        tuned_results[model_name] = search
    return tuned_results


def plot_outputs(
    output_dir: Path,
    y_test,
    baseline_result: dict,
    best_advanced_result: dict,
    all_results: list,
    metrics_table: pd.DataFrame,
) -> None:
    """Generate confusion matrix, ROC comparison, and feature importance plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Confusion matrix for the best advanced model.
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        best_advanced_result["y_pred"],
        display_labels=["No Churn", "Churn"],
        cmap="Blues",
        ax=ax,
    )
    plt.title(f"Confusion Matrix - {best_advanced_result['model']}")
    plt.tight_layout()
    plt.savefig(output_dir / "section4_confusion_matrix_best.png", dpi=300)
    plt.close()

    # Figure 2: ROC curve comparison between baseline and best advanced model.
    fpr_base, tpr_base, _ = roc_curve(y_test, baseline_result["y_prob"])
    fpr_adv, tpr_adv, _ = roc_curve(y_test, best_advanced_result["y_prob"])

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr_base,
        tpr_base,
        linestyle="--",
        label=f"{baseline_result['model']} (AUC={baseline_result['roc_auc']:.3f})",
    )
    plt.plot(
        fpr_adv,
        tpr_adv,
        linewidth=2,
        label=f"{best_advanced_result['model']} (AUC={best_advanced_result['roc_auc']:.3f})",
    )
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "section4_roc_compare.png", dpi=300)
    plt.close()

    # Figure 3: ROC comparison across all models.
    plt.figure(figsize=(8, 6))
    for result in all_results:
        fpr_model, tpr_model, _ = roc_curve(y_test, result["y_prob"])
        plt.plot(
            fpr_model,
            tpr_model,
            linewidth=2,
            label=f"{result['model']} (AUC={result['roc_auc']:.3f})",
        )
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison (All Models)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "section4_roc_all_models.png", dpi=300)
    plt.close()

    # Figure 4: Metric heatmap for all models.
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    heatmap_df = metrics_table.set_index("model")[metric_cols]
    plt.figure(figsize=(9, 5))
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlGnBu", cbar=True)
    plt.title("Model Performance Heatmap")
    plt.xlabel("Metric")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(output_dir / "section4_model_metrics_heatmap.png", dpi=300)
    plt.close()

    # Figure 5: Cross-validation ROC-AUC stability (mean +/- std).
    cv_df = metrics_table[["model", "cv_roc_auc_mean", "cv_roc_auc_std"]].copy()
    plt.figure(figsize=(9, 5))
    x = np.arange(len(cv_df))
    plt.errorbar(
        x,
        cv_df["cv_roc_auc_mean"],
        yerr=cv_df["cv_roc_auc_std"],
        fmt="o",
        capsize=4,
        color="#1f77b4",
    )
    plt.xticks(x, cv_df["model"], rotation=20, ha="right")
    plt.ylabel("CV ROC-AUC (mean ± std)")
    plt.xlabel("Model")
    plt.title("Cross-Validation Stability by Model")
    plt.tight_layout()
    plt.savefig(output_dir / "section4_cv_auc_stability.png", dpi=300)
    plt.close()

    # Figure 6: Feature importance for the best advanced model (tree models only).
    fitted_pipeline = best_advanced_result["fitted_pipeline"]
    model = fitted_pipeline.named_steps["model"]
    preprocessor = fitted_pipeline.named_steps["preprocessor"]

    if hasattr(model, "feature_importances_"):
        feature_names = preprocessor.get_feature_names_out()
        importance = pd.Series(model.feature_importances_, index=feature_names)
        top_features = importance.sort_values(ascending=False).head(20)

        plt.figure(figsize=(10, 7))
        top_features.sort_values().plot(kind="barh", color="teal")
        plt.title(f"Top 20 Feature Importance - {best_advanced_result['model']}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(output_dir / "section4_feature_importance.png", dpi=300)
        plt.close()


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "img" / "ai_data_img"

    X_train, X_test, y_train, y_test = prepare_data(base_dir)
    preprocessor = build_preprocessor(X_train)

    # Baseline model: Logistic Regression.
    baseline_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
            ),
        ]
    )
    baseline_result = evaluate_pipeline(
        name="LogisticRegression",
        pipeline=baseline_pipeline,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    # Advanced models: Decision Tree / Random Forest / Gradient Boosting / Neural Network
    # with hyperparameter tuning.
    tuned_searches = tune_advanced_models(preprocessor, X_train, y_train)

    advanced_results = []
    for model_name, search in tuned_searches.items():
        best_pipeline = search.best_estimator_
        result = evaluate_pipeline(
            name=model_name,
            pipeline=best_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        result["best_params"] = search.best_params_
        advanced_results.append(result)

    # Select the advanced model with the highest test ROC-AUC.
    best_advanced = sorted(advanced_results, key=lambda d: d["roc_auc"], reverse=True)[0]

    # Aggregate and save evaluation metrics.
    rows = [baseline_result] + advanced_results
    metrics_table = pd.DataFrame(
        [
            {
                "model": r["model"],
                "accuracy": r["accuracy"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
                "roc_auc": r["roc_auc"],
                "cv_roc_auc_mean": r["cv_roc_auc_mean"],
                "cv_roc_auc_std": r["cv_roc_auc_std"],
                "best_params": r.get("best_params", {}),
            }
            for r in rows
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_table.to_csv(output_dir / "section4_model_metrics.csv", index=False)

    plot_outputs(
        output_dir=output_dir,
        y_test=y_test,
        baseline_result=baseline_result,
        best_advanced_result=best_advanced,
        all_results=rows,
        metrics_table=metrics_table,
    )

    print("=== Section 4: Modeling and Evaluation Completed ===")
    print(metrics_table.to_string(index=False))
    print(f"\nBest advanced model: {best_advanced['model']}")
    print(f"Best advanced model parameters: {best_advanced['best_params']}")


if __name__ == "__main__":
    main()
