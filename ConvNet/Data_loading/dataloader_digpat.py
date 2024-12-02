import pandas as pd

def get_tile_dataset(csv_file="", valid_labels=None):

    # Assumed that the dataframe contains the columns
    # image_path (absolute path to patch)
    # mask_path (absolute path to binary cell mask)
    # label (...)
    df = pd.read_csv(csv_file).reset_index()

    expected_columns = {"image_path", "mask_path", "label"}

    assert expected_columns.issubset(df.columns), f"file {csv_file} is missing one of image_path, mask_path, label."

    # filter by removing unwanted labels.
    if valid_labels is not None:
        df = df[df["label"].isin(valid_labels)]

    return df