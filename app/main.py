import sys
sys.path.append("src")

from data_loader import load_data
from preprocessing import prepare_features
from preprocessing import split_dataset
from model import train_model
from evaluation import evaluate_model
from visualization import plot_dashboard


def main():
    """
    Main application workflow.
    """

    # Load data
    df = load_data("data/sofia_housing.csv")

    # Prepare data
    X, y, feature_names = prepare_features(df)

    # Split dataset
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    predictions, mae, rmse = evaluate_model(
        model,
        X_test,
        y_test
    )

    # Print results
    print("MAE:", mae)
    print("RMSE:", rmse)

    plot_dashboard(
    df,
    y_test,
    predictions,
    model,
    feature_names
)


if __name__ == "__main__":
    main()