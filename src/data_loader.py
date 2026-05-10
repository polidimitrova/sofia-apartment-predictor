import pandas as pd


def load_data(file_path):
    """
    Loads dataset.
    """

    df = pd.read_csv(
        file_path,
        sep=";"
    )

    return df