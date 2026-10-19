"""Exploratory data analysis: cleaning summary and figures."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from heartml import config, data  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")
LABELS = {0: "No disease", 1: "Disease"}


def run() -> dict:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    raw = data.load_raw()
    df = data.clean(raw)
    data.save_clean(df)

    summary = {
        "rows_raw": int(len(raw)),
        "rows_clean": int(len(df)),
        "duplicates_removed": int(len(raw) - len(df)),
        "missing_values": int(df[config.FEATURES].isna().sum().sum()),
        "class_counts": {LABELS[k]: int(v) for k, v in df[config.TARGET].value_counts().items()},
    }

    # Class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[config.TARGET].map(LABELS).value_counts()
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="deep",
                legend=False, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, str(v), ha="center")
    ax.set_title("Class balance")
    ax.set_ylabel("Patients")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "class_balance.png", dpi=150)
    plt.close(fig)

    # Histograms of numeric features
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, col in zip(axes.ravel(), config.NUMERIC_FEATURES):
        sns.histplot(data=df, x=col, hue=df[config.TARGET].map(LABELS), kde=True,
                     element="step", ax=ax)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Numeric feature distributions by class")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "histograms.png", dpi=150)
    plt.close(fig)

    # Categorical counts by class
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, col in zip(axes.ravel(), config.CATEGORICAL_FEATURES):
        sns.countplot(data=df, x=col, hue=df[config.TARGET].map(LABELS), ax=ax)
        ax.legend(fontsize=7)
    fig.suptitle("Categorical features by class")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "categorical_counts.png", dpi=150)
    plt.close(fig)

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    corr = df.corr()[config.TARGET].drop(config.TARGET).sort_values(key=abs, ascending=False)
    summary["top_correlations_with_target"] = {
        k: round(float(v), 3) for k, v in corr.head(5).items()}

    # Outlier view
    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    for ax, col in zip(axes, config.NUMERIC_FEATURES):
        labels = df[config.TARGET].map(LABELS)
        sns.boxplot(data=df, y=col, x=labels, hue=labels, legend=False, ax=ax)
        ax.set_xlabel("")
    fig.suptitle("Numeric features by class (box plots)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "boxplots.png", dpi=150)
    plt.close(fig)

    (config.ROOT / "reports" / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
