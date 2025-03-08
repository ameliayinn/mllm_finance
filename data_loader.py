import os
import ast
import numpy as np
import pandas as pd

def load_data_from_csv(csv_path):
    """
    Load data from CSV file.

    Args:
    csv_path (str): Path to the CSV file.

    Returns:
    texts (list): List of text data.
    past_arrays (np.ndarray): Array of past time series data.
    future_arrays (np.ndarray): Array of future time series data.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")
    df = pd.read_csv(csv_path)

    texts = []
    past_list = []
    future_list = []

    for idx, row in df.iterrows():
        text_str = str(row["body"])
        past_seq = ast.literal_eval(row["past_50"])
        future_seq = ast.literal_eval(row["future_10"])

        if len(past_seq) != 50:
            raise ValueError(f"Row {idx}: past_50 长度不是 50，实际为 {len(past_seq)}")
        if len(future_seq) != 10:
            raise ValueError(f"Row {idx}: future_10 长度不是 10，实际为 {len(future_seq)}")

        texts.append(text_str)
        past_list.append(past_seq)
        future_list.append(future_seq)

    past_arrays = np.array(past_list, dtype=np.float32)
    future_arrays = np.array(future_list, dtype=np.float32)
    return texts, past_arrays, future_arrays
