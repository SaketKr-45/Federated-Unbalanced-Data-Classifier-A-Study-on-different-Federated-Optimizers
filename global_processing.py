import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler

files = [
    "dataset_random_split1.csv",
    "dataset_random_split2.csv",
    "dataset_random_split3.csv",
    "dataset_random_split4.csv",
]

dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs)

categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(exclude=['object']).drop(columns=['target']).columns

encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
scaler = StandardScaler()

encoder.fit(df[categorical_cols])
scaler.fit(df[numerical_cols])

with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Global preprocessing saved ✅")