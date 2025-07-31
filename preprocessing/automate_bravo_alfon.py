import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def data_preprocessing(csv_path: str, save_path: str):
    """
    Membaca file CSV, melakukan preprocessing (Label Encoding),
    dan menyimpan hasil data train/test ke dalam file terpisah.

    Args:
        csv_path (str): Path ke file CSV mentah
        save_path (str): Folder untuk menyimpan hasil data preprocessing
    """

    df = pd.read_csv(csv_path)

    # Label encode semua kolom kategorikal
    label_encoders = {}
    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Simpan encoder
    os.makedirs("preprocessing/model", exist_ok=True)
    joblib.dump(label_encoders, "preprocessing/model/label_encoders.joblib")

    # Pisahkan fitur dan label
    X = df.drop("class", axis=1)
    y = df["class"]

    # Split train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Simpan hasil
    os.makedirs(save_path, exist_ok=True)
    X_train.to_csv(os.path.join(save_path, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(save_path, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(save_path, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(save_path, "y_test.csv"), index=False)

    print(f"[INFO] Preprocessing selesai. Data disimpan di folder: {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocessing otomatis dataset mushroom.")
    parser.add_argument("--input", type=str, default="mushrooms_raw.csv", help="Path file CSV mentah")
    parser.add_argument("--output", type=str, default="preprocessing/mushrooms_preprocessing", help="Folder hasil preprocessing")
    args = parser.parse_args()

    data_preprocessing(args.input, args.output)
