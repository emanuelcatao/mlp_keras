import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_dataset(file_path):
    return pd.read_csv(file_path)

def select_features(dataset, features):
    return dataset[features]

def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler
