import yfinance as yf
import pandas as pd


class DataLoader:

    def __init__(self, ticker, start="2015-01-01", end="2024-01-01", interval="1d"):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.interval = interval

    def load(self) -> pd.DataFrame:

        try:
            df = yf.download(
                self.ticker,
                start=self.start,
                end=self.end,
                interval=self.interval,
                progress=False,
                threads=False
            )

        except Exception as e:
            print(f"{self.ticker} download error:", e)
            return pd.DataFrame()

        if df is None or len(df) == 0:
            print(f"{self.ticker}: empty dataframe")
            return pd.DataFrame()

        df = df.reset_index()

        df = df[['Date','Open','High','Low','Close','Volume']]
        df.columns = ['date','open','high','low','close','volume']

        print(self.ticker, "rows:", len(df))

        return df