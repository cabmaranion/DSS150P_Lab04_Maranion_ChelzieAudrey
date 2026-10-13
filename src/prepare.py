import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 246112165
DATA = "data/drebin_dataset.csv"
MANIFEST = "metadata/source_manifest.yml"
COL = "TelephonyManager.getSimCountryIso"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_hash(manifest_path):
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("sha256:"):
                return line.split(":", 1)[1].strip()


def load_split():
    if sha256_of(DATA).lower() != expected_hash(MANIFEST).lower():
        raise SystemExit("Data hash does not match the manifest. Stop.")
    print("data hash OK")

    df = pd.read_csv(DATA, dtype={COL: str})
    df[COL] = df[COL].replace("?", "0").astype(int)

    X = df.drop(columns=["class"])
    y = df["class"].map({"S": 1, "B": 0})
    return train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split()
    print("train:", X_train.shape)
    print("test:", X_test.shape)
    print(y_test.value_counts())