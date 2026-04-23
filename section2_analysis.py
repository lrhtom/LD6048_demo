from pathlib import Path

import matplotlib
# Use a non-interactive backend to avoid Tk-related hangs in headless/debug environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Keep plotting style consistent for reproducibility and readability.
RANDOM_STATE = 42
sns.set_theme(style="whitegrid")


def load_and_clean_data(csv_path: Path) -> pd.DataFrame:
    """Load and clean raw data, mainly handling blanks in TotalCharges."""
    df = pd.read_csv(csv_path)

    # Convert blank strings to missing values, then parse as numeric.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop rows missing critical billing values to reduce noise.
    df = df.dropna(subset=["TotalCharges"]).copy()
    return df


def build_summary_tables(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate and save core summary statistic tables."""
    summary = {}

    # Overall churn rate.
    churn_rate = (df["Churn"] == "Yes").mean()
    summary["overall_churn_rate"] = churn_rate

    # Churn rate by contract type.
    contract_churn = (
        df.groupby("Contract")["Churn"]
        .apply(lambda x: (x == "Yes").mean())
        .sort_values(ascending=False)
    )

    # Churn rate by payment method.
    payment_churn = (
        df.groupby("PaymentMethod")["Churn"]
        .apply(lambda x: (x == "Yes").mean())
        .sort_values(ascending=False)
    )

    # Descriptive statistics for numeric features.
    numeric_stats = df[["tenure", "MonthlyCharges", "TotalCharges"]].describe()

    # Save outputs.
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_churn.to_csv(output_dir / "section2_contract_churn_rate.csv", header=["churn_rate"])
    payment_churn.to_csv(output_dir / "section2_payment_churn_rate.csv", header=["churn_rate"])
    numeric_stats.to_csv(output_dir / "section2_numeric_stats.csv")

    print("=== Section 2: Key Statistical Summary ===")
    print(f"Overall churn rate: {churn_rate:.2%}")
    print("\nTop churn rates by contract:")
    print(contract_churn.head(5).to_string())
    print("\nTop churn rates by payment method:")
    print(payment_churn.head(5).to_string())
    print("\nDescriptive statistics for numeric fields:")
    print(numeric_stats.to_string())


def plot_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate report-ready visualization charts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Overall churn distribution.
    plt.figure(figsize=(7, 5))
    order = ["No", "Yes"]
    ax = sns.countplot(data=df, x="Churn", order=order, color="#66c2a5")
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{height / total:.1%}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Customer Count")
    plt.tight_layout()
    plt.savefig(output_dir / "section2_churn_distribution.png", dpi=300)
    plt.close()

    # Figure 2: Churn rate by contract type.
    contract_rate = (
        df.groupby("Contract")["Churn"]
        .apply(lambda x: (x == "Yes").mean())
        .sort_values(ascending=False)
        .reset_index(name="ChurnRate")
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=contract_rate, x="Contract", y="ChurnRate", color="#80b1d3")
    plt.title("Churn Rate by Contract Type")
    plt.xlabel("Contract")
    plt.ylabel("Churn Rate")
    plt.ylim(0, min(1.0, contract_rate["ChurnRate"].max() * 1.2))
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "section2_contract_churn_rate.png", dpi=300)
    plt.close()

    # Figure 3: Churn rate by payment method.
    payment_rate = (
        df.groupby("PaymentMethod")["Churn"]
        .apply(lambda x: (x == "Yes").mean())
        .sort_values(ascending=False)
        .reset_index(name="ChurnRate")
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=payment_rate, x="PaymentMethod", y="ChurnRate", color="#fb8072")
    plt.title("Churn Rate by Payment Method")
    plt.xlabel("Payment Method")
    plt.ylabel("Churn Rate")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "section2_payment_churn_rate.png", dpi=300)
    plt.close()

    # Figure 4: Monthly charge distribution by churn status.
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", order=order, color="#fdb462")
    plt.title("Monthly Charges by Churn Status")
    plt.xlabel("Churn")
    plt.ylabel("MonthlyCharges")
    plt.tight_layout()
    plt.savefig(output_dir / "section2_monthlycharges_by_churn.png", dpi=300)
    plt.close()

    # Figure 5: Tenure distribution by churn status.
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Churn", y="tenure", order=order, color="#8dd3c7")
    plt.title("Tenure by Churn Status")
    plt.xlabel("Churn")
    plt.ylabel("Tenure (months)")
    plt.tight_layout()
    plt.savefig(output_dir / "section2_tenure_by_churn.png", dpi=300)
    plt.close()

    # Figure 6: Correlation heatmap for numeric features.
    plt.figure(figsize=(6, 5))
    corr = df[["tenure", "MonthlyCharges", "TotalCharges"]].corr()
    sns.heatmap(corr, annot=True, cmap="YlGnBu", vmin=-1, vmax=1, square=True)
    plt.title("Numeric Feature Correlation")
    plt.tight_layout()
    plt.savefig(output_dir / "section2_numeric_correlation_heatmap.png", dpi=300)
    plt.close()

    # Figure 7: Descriptive-statistics table rendered as an image.
    stats_table = df[["tenure", "MonthlyCharges", "TotalCharges"]].describe().round(2)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    table = ax.table(
        cellText=stats_table.values,
        rowLabels=stats_table.index,
        colLabels=stats_table.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.2)
    plt.title("Numeric Feature Descriptive Statistics", pad=12)
    plt.tight_layout()
    plt.savefig(output_dir / "section2_numeric_stats_table.png", dpi=300)
    plt.close()


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    output_dir = base_dir / "img" / "data_feathers"

    df = load_and_clean_data(csv_path)
    build_summary_tables(df, output_dir)
    plot_visualizations(df, output_dir)

    print("\nSection 2 analysis completed. Statistical tables and charts have been saved.")


if __name__ == "__main__":
    main()
