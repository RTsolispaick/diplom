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
            pattern_start = -1
            pattern_end = -1

            # Ищем паттерн в текущем окне
            for pattern in patterns:
                p_start = pattern["left_shoulder"]  # конец левого пика
                p_end = pattern["right_shoulder"]    # конец правого пика
                
                # Проверяем, находится ли паттерн в окне
                if p_start >= start and p_end < end:
                    label = 1
                    # Сохраняем координаты относительно начала окна
                    pattern_start = p_start - start
                    pattern_end = p_end - start
                    # Нормализуем на [0, 1]
                    pattern_start = pattern_start / self.window_size
                    pattern_end = pattern_end / self.window_size
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