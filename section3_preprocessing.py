from pathlib import Path

import matplotlib
# Use a non-interactive backend for script-safe plotting.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


sns.set_theme(style="whitegrid")


def load_and_clean_data(csv_path: Path) -> pd.DataFrame:
    """Load and clean data, fixing missing and inconsistent values."""
    df = pd.read_csv(csv_path)

    # TotalCharges contains blank strings; convert to missing values then parse numerically.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop rows with missing critical billing values.
    df = df.dropna(subset=["TotalCharges"]).copy()

    # Remove unique identifier column with no predictive signal.
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features to improve model expressiveness."""
    data = df.copy()

    # Avoid division by zero when tenure equals 0.
    data["AvgChargePerTenure"] = data["TotalCharges"] / (data["tenure"] + 1)

    # Flag new customers (within first 12 months).
    data["IsNewCustomer"] = (data["tenure"] <= 12).astype(int)

    # Aggregate subscribed services as a proxy for customer stickiness.
    service_cols = [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    available_service_cols = [c for c in service_cols if c in data.columns]
    if available_service_cols:
        normalized = data[available_service_cols].replace(
            {
                "No internet service": "No",
                "No phone service": "No",
            }
        )
        data["ServiceCount"] = (normalized == "Yes").sum(axis=1)
    else:
        data["ServiceCount"] = 0

    return data


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a column transformer: scale numerics, one-hot encode categoricals."""
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        # Output dense matrix for direct compatibility with dense-input models such as MLP.
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor


def plot_preprocessing_visualizations(
    df_raw: pd.DataFrame,
    df_processed: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    output_dir: Path,
) -> None:
    """Generate visualization artifacts for each major preprocessing step."""

    # Figure 1: Missing-value handling for TotalCharges.
    raw_missing_totalcharges = pd.to_numeric(df_raw["TotalCharges"], errors="coerce").isna().sum()
    cleaned_missing_totalcharges = df_processed["TotalCharges"].isna().sum()

    plt.figure(figsize=(7, 5))
    missing_df = pd.DataFrame(
        {
            "Stage": ["Before Cleaning", "After Cleaning"],
            "MissingCount": [raw_missing_totalcharges, cleaned_missing_totalcharges],
        }
    )
    ax = sns.barplot(data=missing_df, x="Stage", y="MissingCount", color="#80b1d3")
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.title("Missing Values in TotalCharges (Before vs After Cleaning)")
    plt.xlabel("Preprocessing Stage")
    plt.ylabel("Number of Missing Values")
    plt.tight_layout()
    plt.savefig(output_dir / "section3_missing_value_handling.png", dpi=300)
    plt.close()

    # Figure 2: Class distribution after stratified split.
    split_distribution = pd.DataFrame(
        {
            "Split": ["Train"] * len(y_train) + ["Test"] * len(y_test),
            "Churn": ["Yes" if v == 1 else "No" for v in pd.concat([y_train, y_test], axis=0)],
        }
    )
    plt.figure(figsize=(7, 5))
    sns.countplot(data=split_distribution, x="Split", hue="Churn", palette="Set2")
    plt.title("Class Distribution in Train/Test (Stratified Split)")
    plt.xlabel("Dataset Split")
    plt.ylabel("Customer Count")
    plt.tight_layout()
    plt.savefig(output_dir / "section3_class_distribution_split.png", dpi=300)
    plt.close()

    # Figure 3: Scaling effect for key numeric columns.
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    X_train_scaled_num = scaler.transform(X_train[numeric_features])

    features_to_show = numeric_features[: min(3, len(numeric_features))]
    fig, axes = plt.subplots(len(features_to_show), 2, figsize=(12, 3.8 * len(features_to_show)))

    if len(features_to_show) == 1:
        axes = np.array([axes])

    for i, feature in enumerate(features_to_show):
        sns.histplot(X_train[feature], kde=True, ax=axes[i, 0], color="#fb8072")
        axes[i, 0].set_title(f"Before Scaling: {feature}")
        axes[i, 0].set_xlabel(feature)

        idx = numeric_features.index(feature)
        sns.histplot(X_train_scaled_num[:, idx], kde=True, ax=axes[i, 1], color="#8dd3c7")
        axes[i, 1].set_title(f"After Scaling: {feature}")
        axes[i, 1].set_xlabel(f"Scaled {feature}")

    plt.tight_layout()
    plt.savefig(output_dir / "section3_scaling_before_after.png", dpi=300)
    plt.close()

    # Figure 4: Engineered-feature patterns.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(data=df_processed, x="IsNewCustomer", hue="Churn", ax=axes[0], palette="pastel")
    axes[0].set_title("IsNewCustomer by Churn")
    axes[0].set_xlabel("IsNewCustomer (0/1)")
    axes[0].set_ylabel("Customer Count")

    sns.boxplot(data=df_processed, x="Churn", y="ServiceCount", ax=axes[1], color="#b3de69")
    axes[1].set_title("ServiceCount by Churn")
    axes[1].set_xlabel("Churn")
    axes[1].set_ylabel("ServiceCount")
    plt.tight_layout()
    plt.savefig(output_dir / "section3_feature_engineering_patterns.png", dpi=300)
    plt.close()

    # Figure 5: Dimensionality change after encoding.
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    transformed_feature_count = len(preprocessor.get_feature_names_out())

    dim_df = pd.DataFrame(
        {
            "Stage": ["Numeric Input", "Categorical Input", "After Encoding"],
            "FeatureCount": [len(numeric_features), len(categorical_features), transformed_feature_count],
        }
    )
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=dim_df, x="Stage", y="FeatureCount", color="#fccde5")
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.title("Feature Dimension Change Through Preprocessing")
    plt.xlabel("Pipeline Stage")
    plt.ylabel("Number of Features")
    plt.tight_layout()
    plt.savefig(output_dir / "section3_feature_dimension_change.png", dpi=300)
    plt.close()


def run_preprocessing(base_dir: Path) -> dict:
    """Run full preprocessing pipeline and export intermediate artifacts."""
    csv_path = base_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    output_dir = base_dir / "img" / "data_feathers"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(csv_path)

    df = load_and_clean_data(csv_path)
    df = add_engineered_features(df)

    # Convert target labels: Yes/No -> 1/0.
    y = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    X = df.drop(columns=["Churn"])

    # Use stratified split to preserve class proportions.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    plot_preprocessing_visualizations(
        df_raw=df_raw,
        df_processed=df,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        output_dir=output_dir,
    )

    # Export preprocessing outputs for downstream modeling reuse.
    # Convert to CSR sparse format to support both dense and sparse upstream outputs.
    X_train_csr = sparse.csr_matrix(X_train_processed)
    X_test_csr = sparse.csr_matrix(X_test_processed)

    sparse.save_npz(output_dir / "section3_X_train_processed.npz", X_train_csr)
    sparse.save_npz(output_dir / "section3_X_test_processed.npz", X_test_csr)
    pd.Series(y_train).to_csv(output_dir / "section3_y_train.csv", index=False)
    pd.Series(y_test).to_csv(output_dir / "section3_y_test.csv", index=False)
    joblib.dump(preprocessor, output_dir / "section3_preprocessor.joblib")

    # Save feature names for model interpretation.
    feature_names = preprocessor.get_feature_names_out()
    pd.Series(feature_names, name="feature_name").to_csv(
        output_dir / "section3_feature_names.csv", index=False
    )

    result = {
        "X_train_shape": X_train_processed.shape,
        "X_test_shape": X_test_processed.shape,
        "y_train_positive_rate": float(y_train.mean()),
        "y_test_positive_rate": float(y_test.mean()),
        "feature_count": len(feature_names),
    }
    return result


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    info = run_preprocessing(base_dir)

    print("=== Section 3: Preprocessing Completed ===")
    print(f"Training set shape: {info['X_train_shape']}")
    print(f"Test set shape: {info['X_test_shape']}")
    print(f"Training churn rate: {info['y_train_positive_rate']:.2%}")
    print(f"Test churn rate: {info['y_test_positive_rate']:.2%}")
    print(f"Number of processed features: {info['feature_count']}")
    print("Preprocessing visual diagnostics saved under img/data_feathers.")


if __name__ == "__main__":
    main()
