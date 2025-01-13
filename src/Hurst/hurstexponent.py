import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import yfinance as yf
from typing import Tuple, Union
import warnings


class HurstAnalyzer:

    def __init__(self, min_window: int = 10):
        self.min_window = min_window

    def _calculate_rs(self, data: np.ndarray, window: int) -> float:

        try:

            series = data[:window]
            mean = np.mean(series)
            dev = series - mean
            cumdev = np.cumsum(dev)

            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(series)

            if S == 0:
                return np.nan

            return R / S

        except Exception:
            return np.nan

    def calculate_hurst(self, data: Union[list, np.ndarray, pd.Series]) -> Tuple[float, float]:

        data = np.array(data, dtype=float)
        data = data[~np.isnan(data)]

        if len(data) < self.min_window:
            raise ValueError(f"Data length must be at least {self.min_window}")


        max_window = len(data) // 2
        window_sizes = np.logspace(
            np.log10(self.min_window),
            np.log10(max_window),
            num=20,
            dtype=int
        )
        window_sizes = np.unique(window_sizes)

        rs_values = []
        for w in window_sizes:
            rs = self._calculate_rs(data, w)
            if not np.isnan(rs) and rs > 0:
                rs_values.append(rs)
            else:
                rs_values.append(np.nan)

        rs_values = np.array(rs_values)

        valid_mask = ~np.isnan(rs_values)
        rs_values = rs_values[valid_mask]
        window_sizes = window_sizes[valid_mask]

        if len(window_sizes) < 4: #Minimum condition, otherwise the code breaks.
            raise ValueError("Insufficient valid data points ")

        # Perform linear regression but log sclae
        log_window = np.log10(window_sizes)
        log_rs = np.log10(rs_values)

        slope, intercept, r_value, p_value, stderr = stats.linregress(log_window, log_rs)

        return slope, r_value ** 2

    def analyze_financial(self, symbol: str, start_date: str, end_date: str = None) -> None:

        # Fetch data
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)

        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")


        close_prices = data['Close'].values
        returns = np.diff(np.log(close_prices))

        hurst, r_squared = self.calculate_hurst(returns)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                name='Price',
                mode='lines',
                line=dict(color='rgba(0, 123, 255, 0.8)', width=2)
            )
        )

        X = np.arange(len(data))
        model = np.polyfit(X, data['Close'], 1)  # Linear fit
        trendline = np.poly1d(model)
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=trendline(X),
                name='Trendline',
                line=dict(color='rgba(48, 120, 71, 0.8)', dash='dash', width=2)
            )
        )

        type_strat = "Mean reverting series" if hurst < 0.5 else "Trending series" if hurst > 0.5 else "Brownian Motion"
        fig.add_annotation(
            text=f"<b>Hurst Exponent: {hurst:.3f} </b><br><b>R²: {r_squared:.3f}</b><br><b>{type_strat}</b>",
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            showarrow=False,
            font=dict(size=12, color="black"),
            bordercolor="rgba(0, 0, 0, 0.9)",
            borderwidth=1,
            borderpad=5,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )

        fig.update_layout(
            title=f"<b>{symbol} Price Analysis with Hurst Exponent</b>",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=12, color="rgba(0, 0, 0, 0.9)"),
            height=600,
            width=1100,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(230, 230, 230, 0.5)"
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(230, 230, 230, 0.5)"
            )
        )

        fig.show()

        print(f"\nHurst Exponent: {hurst:.3f}")
        print(f"R-squared: {r_squared:.3f}")
        print("\nInterpretation:")
        if hurst < 0.45:
            print("Mean-reverting (anti-persistent) series")
        elif hurst > 0.55:
            print("Trending (persistent) series")
        else:
            print("Random walk (Brownian motion)")