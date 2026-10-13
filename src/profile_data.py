import pandas as pd

df = pd.read_csv("data/source/drebin_dataset.csv", dtype={"TelephonyManager.getSimCountryIso": str})

print("rows, columns:", df.shape)
print("class counts:")
print(df["class"].value_counts())
print("total missing values:", df.isna().sum().sum())
print("exact duplicate rows:", df.duplicated().sum())

print("columns that are not plain 0/1:")
for col in df.columns:
    values = set(df[col].unique())
    if col != "class" and not values.issubset({0, 1}):
        print(" ", col, values)

features = df.drop(columns="class")
print("duplicates counting class:", df.duplicated().sum())
print("duplicates ignoring class:", features.duplicated().sum())