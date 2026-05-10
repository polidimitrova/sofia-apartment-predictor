import tkinter as tk
from tkinter import ttk
import pandas as pd

from data_loader import load_data
from preprocessing import prepare_features
from model import train_model


# load Sofia dataset
df = load_data("data/sofia_housing.csv")

# train model once
X, y, feature_names = prepare_features(df)
model = train_model(X, y)


districts = sorted(df["district"].unique())
building_types = sorted(df["building_type"].unique())


def predict_price():
    area = float(area_entry.get())
    bedrooms = float(bedrooms_entry.get())
    bathrooms = float(bathrooms_entry.get())
    floor = float(floor_entry.get())

    district = district_var.get()
    building_type = building_var.get()

    input_data = pd.DataFrame({
        "area_m2": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "floor": [floor],
        "district": [district],
        "building_type": [building_type]
    })

    X_input, _, _ = prepare_features(
        pd.concat([df, input_data], ignore_index=True)
    )

    prediction = model.predict(
        X_input[-1:].reshape(1, -1)
    )[0]

    result_label.config(
        text=f"Estimated price in {district}: €{prediction:,.0f}"
    )



root = tk.Tk()
root.title("Sofia Apartment Predictor")
root.geometry("420x520")


tk.Label(root, text="Area (m²)").pack()
area_entry = tk.Entry(root)
area_entry.pack()


tk.Label(root, text="Bedrooms").pack()
bedrooms_entry = tk.Entry(root)
bedrooms_entry.pack()


tk.Label(root, text="Bathrooms").pack()
bathrooms_entry = tk.Entry(root)
bathrooms_entry.pack()


tk.Label(root, text="Floor").pack()
floor_entry = tk.Entry(root)
floor_entry.pack()


tk.Label(root, text="District").pack()
district_var = tk.StringVar()
district_menu = ttk.Combobox(
    root,
    textvariable=district_var,
    values=districts,
    state="readonly"
)
district_menu.current(0)
district_menu.pack()


tk.Label(root, text="Building Type").pack()
building_var = tk.StringVar()
building_menu = ttk.Combobox(
    root,
    textvariable=building_var,
    values=building_types,
    state="readonly"
)
building_menu.current(0)
building_menu.pack()


tk.Button(
    root,
    text="Predict Price",
    command=predict_price
).pack(pady=20)


result_label = tk.Label(
    root,
    text="",
    font=("Arial", 14)
)
result_label.pack()


root.mainloop()