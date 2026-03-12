import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


class HeadShouldersDetector:

    def __init__(
        self,
        order=5,
        shoulder_tol=0.25,
        min_head_height=0.02,
        min_distance=8,
        max_width=120
    ):
        self.order = order
        self.shoulder_tol = shoulder_tol
        self.min_head_height = min_head_height
        self.min_distance = min_distance
        self.max_width = max_width

    def _find_extrema(self, prices):
        peaks = argrelextrema(prices, np.greater, order=self.order)[0]
        troughs = argrelextrema(prices, np.less, order=self.order)[0]

        extrema = []

        for p in peaks:
            extrema.append((p, "peak"))

        for t in troughs:
            extrema.append((t, "trough"))

        extrema.sort(key=lambda x: x[0])

        return extrema

    def find_patterns(self, df: pd.DataFrame, inverse=False):
        prices = df["close"].values
        extrema = self._find_extrema(prices)

        patterns = []

        for i in range(len(extrema) - 4):
            idx1, t1 = extrema[i]
            idx2, t2 = extrema[i + 1]
            idx3, t3 = extrema[i + 2]
            idx4, t4 = extrema[i + 3]
            idx5, t5 = extrema[i + 4]

            if not inverse:
                if not (t1=="peak" and t2=="trough" and t3=="peak" and t4=="trough" and t5=="peak"):
                    continue

                ls = prices[idx1]
                head = prices[idx3]
                rs = prices[idx5]

                if not (head > ls and head > rs):
                    continue

                avg_shoulder = (ls + rs) / 2

                if (head - avg_shoulder) / avg_shoulder < self.min_head_height:
                    continue

                if abs(ls - rs) / avg_shoulder > self.shoulder_tol:
                    continue
            else:
                if not (t1=="trough" and t2=="peak" and t3=="trough" and t4=="peak" and t5=="trough"):
                    continue

                ls = prices[idx1]
                head = prices[idx3]
                rs = prices[idx5]

                if not (head < ls and head < rs):
                    continue

                avg_shoulder = (ls + rs) / 2

                if (avg_shoulder - head) / avg_shoulder < self.min_head_height:
                    continue

                if abs(ls - rs) / avg_shoulder > self.shoulder_tol:
                    continue

            if idx5 - idx1 > self.max_width:
                continue

            if idx3 - idx1 < self.min_distance:
                continue

            if idx5 - idx3 < self.min_distance:
                continue

            patterns.append(
                {
                    "left_shoulder": idx1,
                    "head": idx3,
                    "right_shoulder": idx5,
                    "trough1": idx2,
                    "trough2": idx4,
                    "type": "inverse" if inverse else "normal"
                }
            )

        return patterns