from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def prepare_features(df):

    numeric_features = df[
        [
            "area_m2",
            "bedrooms",
            "bathrooms",
            "floor"
        ]
    ]


    categorical_features = pd.get_dummies(
        df[
            [
                "district",
                "building_type"
            ]
        ]
    )


    X = pd.concat(
        [
            numeric_features,
            categorical_features
        ],
        axis=1
    )


    y = df["price_eur"]


    feature_names = X.columns
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, feature_names
    

def split_dataset(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test