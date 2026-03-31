import numpy as np
import pandas as pd


class DatasetBuilder:

    def __init__(self, window_size=120, step=5):
        self.window_size = window_size
        self.step = step

    def build(self, df, patterns):
        prices = df["close"].values

        X = []
        y_labels = []
        y_coords = []

        for start in range(0, len(prices) - self.window_size, self.step):
            end = start + self.window_size
            window = prices[start:end]

            label = 0
            pattern_start = 0.0
            pattern_end = 0.0

            # Ищем паттерн в текущем окне
            for pattern in patterns:
                p_start = pattern["left_shoulder"]
                p_end = pattern["right_shoulder"]

                # Проверяем, находится ли паттерн полностью в окне
                if p_start >= start and p_end < end:
                    label = 1
                    # Сохраняем координаты относительно начала окна
                    pattern_start = (p_start - start) / max(self.window_size - 1, 1)
                    pattern_end = (p_end - start) / max(self.window_size - 1, 1)
                    break

            std = window.std()

            if std == 0:
                window = window - window.mean()
            else:
                window = (window - window.mean()) / std

            X.append(window)
            y_labels.append(label)
            y_coords.append([label, pattern_start, pattern_end])

        return np.array(X), np.array(y_labels), np.array(y_coords)