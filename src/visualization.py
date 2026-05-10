import matplotlib.pyplot as plt
plt.rcParams["toolbar"] = "None"

def plot_predictions(y_test, predictions):
    plt.figure(figsize=(8, 5))

    # convert to millions
    y_test_m = y_test / 1_000_000
    predictions_m = predictions / 1_000_000

    plt.scatter(y_test_m, predictions_m)

    plt.xlabel("Actual Prices (Millions)")
    plt.ylabel("Predicted Prices (Millions)")
    plt.title("Actual vs Predicted House Prices")

    plt.show()

def plot_feature_importance(model, feature_names):
    import matplotlib.pyplot as plt

    import numpy as np
    coefficients = np.abs(model.coef_)
    coefficients = coefficients / coefficients.max()

    plt.figure(figsize=(9, 5))

    bars = plt.bar(feature_names, coefficients)

    plt.title("Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Coefficient (Millions)")

    plt.xticks(rotation=15)

    # value labels
    for bar in bars:
        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom"
        )

    plt.savefig("images/feature_importance.png")

    plt.show()

def plot_residuals(y_test, predictions):
    import matplotlib.pyplot as plt

    residuals = y_test - predictions

    plt.figure(figsize=(8, 5))

    plt.scatter(predictions, residuals)

    plt.axhline(y=0, linestyle="--")

    plt.title("Residual Analysis")
    plt.xlabel("Predicted Prices")
    plt.ylabel("Residuals")

    plt.savefig("images/residuals.png")
    fig.canvas.manager.set_window_title(
    "House Price Prediction Dashboard"
    )
    plt.show()


def plot_dashboard(df, y_test, predictions, model, feature_names):
    import matplotlib.pyplot as plt

    residuals = y_test - predictions

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 8)
    )
    plt.gcf().canvas.manager.set_window_title(
        "House Price Prediction Dashboard"
    )
    # 1. Predictions
    axes[0, 0].scatter(
        y_test / 1_000_000,
        predictions / 1_000_000
    )

    axes[0,0].set_title("Actual vs Predicted Prices")
    axes[0, 0].set_xlabel("Actual (M)")
    axes[0, 0].set_ylabel("Predicted (M)")


    # 2. Feature importance
    import numpy as np
    coefficients = np.abs(model.coef_)
    coefficients = coefficients / coefficients.max()

    coefficients = abs(model.coef_)
    feature_groups = {
        "Area": 0,
        "Bedrooms": 0,
        "Bathrooms": 0,
        "Floor": 0,
        "District": 0,
        "Building Type": 0
    }
    for i, name in enumerate(feature_names):
        if "area" in name:
            feature_groups["Area"] += coefficients[i]
        
        elif "bedrooms" in name:
            feature_groups["Bedrooms"] += coefficients[i]
        
        elif "bathrooms" in name:
            feature_groups["Bathrooms"] += coefficients[i]
        
        elif "floor" in name:
            feature_groups["Floor"] += coefficients[i]
        
        elif "district" in name:
            feature_groups["District"] += coefficients[i]
        
        elif "building" in name:
            feature_groups["Building Type"] += coefficients[i]
        
    
    axes[0, 1].bar(
        list(feature_groups.keys()),
        list(feature_groups.values())
    )
    axes[0, 1].set_title("Feature Importance")
    axes[0, 1].tick_params(
        axis="x",
        rotation=35,
        labelsize=9
    )


    # 3. Residuals
    axes[1, 0].scatter(
        predictions / 1_000_000,
        residuals / 1_000_000
    )

    axes[1, 0].axhline(y=0, linestyle="--")

    axes[1, 0].set_title("Residual Analysis")
    axes[1, 0].set_xlabel("Predicted (M)")
    axes[1, 0].set_ylabel("Residuals (M)")


    # 4. Correlation matrix
    correlation = df.corr(numeric_only=True)

    heatmap = axes[1, 1].imshow(correlation)

    axes[1, 1].set_title("Correlation Matrix")

    axes[1, 1].set_xticks(range(len(correlation.columns)))
    axes[1, 1].set_xticklabels(
        correlation.columns,
        rotation=45
    )

    axes[1, 1].set_yticks(range(len(correlation.columns)))
    axes[1, 1].set_yticklabels(
        correlation.columns
    )

    fig.colorbar(
        heatmap,
        ax=axes[1, 1]
    )


    plt.tight_layout()
    fig.savefig("images/dashboard.png")
    return fig

