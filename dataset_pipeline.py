import pandas as pd
import numpy as np

from data_loader import DataLoader
from pattern_detector import HeadShouldersDetector
from dataset_builder import DatasetBuilder


def normalize_window(window):
    std = np.std(window)
    if std == 0:
        return window
    return (window - np.mean(window)) / std


def build_dataset_for_ticker(ticker, window_size=120):
    loader = DataLoader(ticker)
    df = loader.load()

    detector = HeadShouldersDetector()
    patterns = detector.find_patterns(df)

    builder = DatasetBuilder(window_size=window_size)

    X, y_labels, y_coords = builder.build(df, patterns)

    X = np.array([normalize_window(w) for w in X])

    return X, y_labels, y_coords


def save_csv(X, y_coords, filename):
    window_size = X.shape[1]

    columns = [f"t{i}" for i in range(window_size)]

    df = pd.DataFrame(X, columns=columns)
    df["has_pattern"] = y_coords[:, 0].astype(int)
    df["pattern_start"] = y_coords[:, 1]
    df["pattern_end"] = y_coords[:, 2]

    df.to_csv(filename, index=False)