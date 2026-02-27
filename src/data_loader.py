import pandas as pd


def load_dataset(data_path, label_path):
    df = pd.read_csv(data_path)

    df_clean = df.drop(
        columns=[col for col in df.columns if "call" in col] +
        ["Gene Description"]
    )

    df_trans = df_clean.set_index(
        "Gene Accession Number"
    ).T.reset_index().rename(columns={"index": "patient"})

    df_trans["patient"] = df_trans["patient"].astype(int)

    labels = pd.read_csv(label_path)
    labels["patient"] = labels["patient"].astype(int)

    final_dataset = df_trans.merge(labels, on="patient")

    X = final_dataset.drop("cancer", axis=1)
    y = final_dataset["cancer"]

    return X, y
