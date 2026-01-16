"""
Market Data Visualization Module.

Provides specialized charts for financial market data including:
- Candlestick/OHLC charts with volume
- Price line charts with multiple symbols
- Spread visualization over time
- Quote imbalance heatmaps
- Trade imbalance visualization
"""

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .charts import BaseChart, MultiPanelChart
from .themes import ThemeConfig, get_theme


class CandlestickChart(MultiPanelChart):
    """
    Candlestick/OHLC chart with optional volume panel.

    Provides interactive financial charts with configurable overlays
    and indicators.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        title: str = "",
        width: int = 1200,
        height: int = 700,
        theme: ThemeConfig | None = None,
        date_col: str = "bar_start",
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str | None = "volume",
        show_volume: bool = True,
    ) -> None:
        """
        Initialize candlestick chart.

        Args:
            data: DataFrame with OHLCV data.
            title: Chart title.
            width: Chart width in pixels.
            height: Chart height in pixels.
            theme: Optional theme configuration.
            date_col: Column name for datetime.
            open_col: Column name for open price.
            high_col: Column name for high price.
            low_col: Column name for low price.
            close_col: Column name for close price.
            volume_col: Column name for volume (optional).
            show_volume: Whether to show volume panel.
        """
        self.date_col = date_col
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        self.show_volume = show_volume and volume_col is not None

        rows = 2 if self.show_volume else 1
        row_heights = [0.7, 0.3] if self.show_volume else [1.0]

        super().__init__(
            data=data,
            title=title,
            width=width,
            height=height,
            theme=theme,
            rows=rows,
            row_heights=row_heights,
            shared_xaxes=True,
            vertical_spacing=0.05,
        )

    def _create_figure(self) -> go.Figure:
        """Create candlestick chart with optional volume."""
        fig = super()._create_figure()

        if self.data is None:
            return fig

        theme = self.theme

        # Add candlestick trace
        fig.add_trace(
            go.Candlestick(
                x=self.data[self.date_col],
                open=self.data[self.open_col],
                high=self.data[self.high_col],
                low=self.data[self.low_col],
                close=self.data[self.close_col],
                name="OHLC",
                increasing={"line": {"color": theme.colors.bullish}},
                decreasing={"line": {"color": theme.colors.bearish}},
            ),
            row=1,
            col=1,
        )

        # Add volume if requested
        if self.show_volume and self.volume_col:
            colors = [
                theme.colors.volume_up
                if self.data[self.close_col].iloc[i] >= self.data[self.open_col].iloc[i]
                else theme.colors.volume_down
                for i in range(len(self.data))
            ]

            fig.add_trace(
                go.Bar(
                    x=self.data[self.date_col],
                    y=self.data[self.volume_col],
                    name="Volume",
                    marker_color=colors,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

            fig.update_yaxes(title_text="Volume", row=2, col=1)

        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_xaxes(rangeslider_visible=False)

        return fig

    def add_moving_average(
        self,
        period: int,
        column: str | None = None,
        color: str | None = None,
        name: str | None = None,
    ) -> "CandlestickChart":
        """Add moving average line overlay."""
        if self.data is None:
            return self

        col = column or self.close_col
        ma = self.data[col].rolling(window=period).mean()
        line_color = color or self.theme.colors.secondary
        line_name = name or f"MA({period})"

        self.figure.add_trace(
            go.Scatter(
                x=self.data[self.date_col],
                y=ma,
                name=line_name,
                mode="lines",
                line={"color": line_color, "width": 1},
            ),
            row=1,
            col=1,
        )
        return self

    def add_bollinger_bands(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        column: str | None = None,
        color: str | None = None,
    ) -> "CandlestickChart":
        """Add Bollinger Bands overlay."""
        if self.data is None:
            return self

        col = column or self.close_col
        ma = self.data[col].rolling(window=period).mean()
        std = self.data[col].rolling(window=period).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)

        line_color = color or self.theme.colors.info

        # Upper band
        self.figure.add_trace(
            go.Scatter(
                x=self.data[self.date_col],
                y=upper,
                name=f"BB Upper ({period}, {std_dev})",
                mode="lines",
                line={"color": line_color, "width": 1, "dash": "dot"},
            ),
            row=1,
            col=1,
        )

        # Lower band
        self.figure.add_trace(
            go.Scatter(
                x=self.data[self.date_col],
                y=lower,
                name=f"BB Lower ({period}, {std_dev})",
                mode="lines",
                line={"color": line_color, "width": 1, "dash": "dot"},
                fill="tonexty",
                fillcolor=f"rgba{tuple(list(int(line_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.1])}",
            ),
            row=1,
            col=1,
        )

        return self


class MarketChart(MultiPanelChart):
    """
    Comprehensive market chart with multiple panels.

    Supports candlesticks, volume, spread, and other overlays
    in a single coordinated view.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        title: str = "",
        width: int = 1200,
        height: int = 900,
        theme: ThemeConfig | None = None,
        date_col: str = "bar_start",
    ) -> None:
        """
        Initialize market chart.

        Args:
            data: DataFrame with market data.
            title: Chart title.
            width: Chart width in pixels.
            height: Chart height in pixels.
            theme: Optional theme configuration.
            date_col: Column name for datetime.
        """
        self.date_col = date_col
        self._panels: list[str] = []
        super().__init__(
            data=data,
            title=title,
            width=width,
            height=height,
            theme=theme,
            rows=1,
            row_heights=[1.0],
            shared_xaxes=True,
        )

    def _create_figure(self) -> go.Figure:
        """Create empty figure - panels added via methods."""
        fig = go.Figure()

        layout = self.theme.to_plotly_layout()
        layout.update(
            title=self.title,
            width=self.width,
            height=self.height,
            showlegend=True,
            legend={"x": 0, "y": 1.02, "orientation": "h"},
            xaxis={"rangeslider": {"visible": False}},
        )
        fig.update_layout(**layout)

        return fig

    def _rebuild_subplots(self) -> None:
        """Rebuild the figure with updated panel count."""
        n_panels = max(1, len(self._panels))
        heights = [0.5] + [0.5 / max(1, n_panels - 1)] * max(0, n_panels - 1)
        if n_panels == 1:
            heights = [1.0]

        self._figure = make_subplots(
            rows=n_panels,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=heights,
        )

        layout = self.theme.to_plotly_layout()
        layout.update(
            title=self.title,
            width=self.width,
            height=self.height,
            showlegend=True,
            legend={"x": 0, "y": 1.02, "orientation": "h"},
        )
        self._figure.update_layout(**layout)

    def add_candlesticks(
        self,
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
    ) -> "MarketChart":
        """Add candlestick panel."""
        if self.data is None:
            return self

        if "price" not in self._panels:
            self._panels.insert(0, "price")
            self._rebuild_subplots()

        row = self._panels.index("price") + 1
        theme = self.theme

        self.figure.add_trace(
            go.Candlestick(
                x=self.data[self.date_col],
                open=self.data[open_col],
                high=self.data[high_col],
                low=self.data[low_col],
                close=self.data[close_col],
                name="OHLC",
                increasing={"line": {"color": theme.colors.bullish}},
                decreasing={"line": {"color": theme.colors.bearish}},
            ),
            row=row,
            col=1,
        )

        self.figure.update_yaxes(title_text="Price", row=row, col=1)
        self.figure.update_xaxes(rangeslider_visible=False)

        return self

    def add_volume(self, volume_col: str = "volume") -> "MarketChart":
        """Add volume panel."""
        if self.data is None or volume_col not in self.data.columns:
            return self

        if "volume" not in self._panels:
            self._panels.append("volume")
            self._rebuild_subplots()

        row = self._panels.index("volume") + 1
        theme = self.theme

        # Determine color based on price direction if possible
        if "close" in self.data.columns and "open" in self.data.columns:
            colors = [
                theme.colors.volume_up if c >= o else theme.colors.volume_down
                for c, o in zip(self.data["close"], self.data["open"], strict=False)
            ]
        else:
            colors = [theme.colors.primary] * len(self.data)

        self.figure.add_trace(
            go.Bar(
                x=self.data[self.date_col],
                y=self.data[volume_col],
                name="Volume",
                marker_color=colors,
                showlegend=False,
            ),
            row=row,
            col=1,
        )

        self.figure.update_yaxes(title_text="Volume", row=row, col=1)
        return self

    def add_spread_overlay(
        self,
        spread_col: str = "spread",
        secondary_y: bool = True,
    ) -> "MarketChart":
        """Add spread as overlay or separate panel."""
        if self.data is None or spread_col not in self.data.columns:
            return self

        if "spread" not in self._panels:
            self._panels.append("spread")
            self._rebuild_subplots()

        row = self._panels.index("spread") + 1

        self.figure.add_trace(
            go.Scatter(
                x=self.data[self.date_col],
                y=self.data[spread_col],
                name="Spread",
                mode="lines",
                line={"color": self.theme.colors.warning, "width": 1},
                fill="tozeroy",
                fillcolor="rgba(255, 187, 120, 0.3)",
            ),
            row=row,
            col=1,
        )

        self.figure.update_yaxes(title_text="Spread", row=row, col=1)
        return self

    def add_price_line(
        self,
        price_col: str,
        name: str | None = None,
        color: str | None = None,
    ) -> "MarketChart":
        """Add price line to the price panel."""
        if self.data is None or price_col not in self.data.columns:
            return self

        if "price" not in self._panels:
            self._panels.insert(0, "price")
            self._rebuild_subplots()

        row = self._panels.index("price") + 1

        self.figure.add_trace(
            go.Scatter(
                x=self.data[self.date_col],
                y=self.data[price_col],
                name=name or price_col,
                mode="lines",
                line={"color": color or self.theme.colors.secondary, "width": 1.5},
            ),
            row=row,
            col=1,
        )

        return self


class PriceLineChart(BaseChart):
    """
    Multi-symbol price line chart.

    Visualizes price trends for multiple symbols on the same chart
    with optional normalization for comparison.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        date_col: str = "bar_start",
        price_cols: list[str] | None = None,
        symbol_col: str | None = None,
        title: str = "Price Comparison",
        width: int = 1200,
        height: int = 600,
        theme: ThemeConfig | None = None,
        normalize: bool = False,
    ) -> None:
        """
        Initialize price line chart.

        Args:
            data: DataFrame with price data.
            date_col: Column name for datetime.
            price_cols: List of columns to plot (if data is wide format).
            symbol_col: Column name for symbol (if data is long format).
            title: Chart title.
            width: Chart width in pixels.
            height: Chart height in pixels.
            theme: Optional theme configuration.
            normalize: Whether to normalize prices to 100 at start.
        """
        self.date_col = date_col
        self.price_cols = price_cols
        self.symbol_col = symbol_col
        self.normalize = normalize
        super().__init__(data, title, width, height, theme)

    def _create_figure(self) -> go.Figure:
        """Create price line chart."""
        fig = go.Figure()

        if self.data is None:
            return fig

        colors = [
            self.theme.colors.primary,
            self.theme.colors.secondary,
            self.theme.colors.success,
            self.theme.colors.danger,
            self.theme.colors.warning,
            self.theme.colors.info,
        ]

        # Handle wide format (multiple price columns)
        if self.price_cols:
            for i, col in enumerate(self.price_cols):
                if col not in self.data.columns:
                    continue

                y_values = self.data[col].copy()
                if self.normalize and len(y_values) > 0:
                    first_valid = y_values.dropna().iloc[0] if not y_values.dropna().empty else 1
                    y_values = (y_values / first_valid) * 100

                fig.add_trace(
                    go.Scatter(
                        x=self.data[self.date_col],
                        y=y_values,
                        name=col,
                        mode="lines",
                        line={"color": colors[i % len(colors)], "width": self.theme.line_width},
                    )
                )

        # Handle long format (symbol column)
        elif self.symbol_col and self.symbol_col in self.data.columns:
            symbols = self.data[self.symbol_col].unique()
            for i, symbol in enumerate(symbols):
                mask = self.data[self.symbol_col] == symbol
                subset = self.data[mask]

                # Find the price column (could be close, price, etc.)
                price_col = next(
                    (c for c in ["close", "price", "midprice"] if c in subset.columns), None
                )
                if price_col is None:
                    continue

                y_values = subset[price_col].copy()
                if self.normalize and len(y_values) > 0:
                    first_valid = y_values.dropna().iloc[0] if not y_values.dropna().empty else 1
                    y_values = (y_values / first_valid) * 100

                fig.add_trace(
                    go.Scatter(
                        x=subset[self.date_col],
                        y=y_values,
                        name=str(symbol),
                        mode="lines",
                        line={"color": colors[i % len(colors)], "width": self.theme.line_width},
                    )
                )

        # Apply theme
        layout = self.theme.to_plotly_layout()
        y_title = "Normalized Price (%)" if self.normalize else "Price"
        layout.update(
            title=self.title,
            width=self.width,
            height=self.height,
            showlegend=True,
            yaxis_title=y_title,
        )
        fig.update_layout(**layout)

        return fig


class SpreadChart(BaseChart):
    """
    Bid-ask spread visualization over time.

    Shows spread evolution with statistical bands and highlights.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        date_col: str = "bar_start",
        spread_col: str = "spread",
        title: str = "Spread Over Time",
        width: int = 1200,
        height: int = 500,
        theme: ThemeConfig | None = None,
        show_rolling_mean: bool = True,
        rolling_window: int = 20,
    ) -> None:
        """
        Initialize spread chart.

        Args:
            data: DataFrame with spread data.
            date_col: Column name for datetime.
            spread_col: Column name for spread values.
            title: Chart title.
            width: Chart width in pixels.
            height: Chart height in pixels.
            theme: Optional theme configuration.
            show_rolling_mean: Whether to show rolling mean.
            rolling_window: Window size for rolling statistics.
        """
        self.date_col = date_col
        self.spread_col = spread_col
        self.show_rolling_mean = show_rolling_mean
        self.rolling_window = rolling_window
        super().__init__(data, title, width, height, theme)

    def _create_figure(self) -> go.Figure:
        """Create spread chart."""
        fig = go.Figure()

        if self.data is None or self.spread_col not in self.data.columns:
            return fig

        # Raw spread
        fig.add_trace(
            go.Scatter(
                x=self.data[self.date_col],
                y=self.data[self.spread_col],
                name="Spread",
                mode="lines",
                line={"color": self.theme.colors.primary, "width": 1},
                fill="tozeroy",
                fillcolor="rgba(31, 119, 180, 0.2)",
            )
        )

        # Rolling mean
        if self.show_rolling_mean:
            rolling_mean = self.data[self.spread_col].rolling(window=self.rolling_window).mean()
            fig.add_trace(
                go.Scatter(
                    x=self.data[self.date_col],
                    y=rolling_mean,
                    name=f"MA({self.rolling_window})",
                    mode="lines",
                    line={"color": self.theme.colors.secondary, "width": 2},
                )
            )

        # Apply theme
        layout = self.theme.to_plotly_layout()
        layout.update(
            title=self.title,
            width=self.width,
            height=self.height,
            showlegend=True,
            yaxis_title="Spread",
        )
        fig.update_layout(**layout)

        return fig


class QuoteImbalanceHeatmap(BaseChart):
    """
    Quote imbalance heatmap visualization.

    Shows imbalance patterns over time with color-coded intensity.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        date_col: str = "bar_start",
        imbalance_col: str = "quote_imbalance",
        symbol_col: str | None = "symbol",
        title: str = "Quote Imbalance Heatmap",
        width: int = 1200,
        height: int = 600,
        theme: ThemeConfig | None = None,
    ) -> None:
        """
        Initialize quote imbalance heatmap.

        Args:
            data: DataFrame with imbalance data.
            date_col: Column name for datetime.
            imbalance_col: Column name for imbalance values.
            symbol_col: Column name for symbols (creates rows).
            title: Chart title.
            width: Chart width in pixels.
            height: Chart height in pixels.
            theme: Optional theme configuration.
        """
        self.date_col = date_col
        self.imbalance_col = imbalance_col
        self.symbol_col = symbol_col
        super().__init__(data, title, width, height, theme)

    def _create_figure(self) -> go.Figure:
        """Create quote imbalance heatmap."""
        fig = go.Figure()

        if self.data is None or self.imbalance_col not in self.data.columns:
            return fig

        # Pivot data if we have symbols
        if self.symbol_col and self.symbol_col in self.data.columns:
            pivot = self.data.pivot_table(
                index=self.symbol_col,
                columns=self.date_col,
                values=self.imbalance_col,
                aggfunc="mean",
            )
            z = pivot.values
            x = list(pivot.columns)
            y = list(pivot.index)
        else:
            # Single symbol - reshape for heatmap
            z = self.data[self.imbalance_col].values.reshape(1, -1)
            x = list(self.data[self.date_col])
            y = ["Imbalance"]

        # Diverging colorscale centered at 0
        colorscale = [
            [0.0, self.theme.colors.bearish],
            [0.5, self.theme.colors.neutral],
            [1.0, self.theme.colors.bullish],
        ]

        fig.add_trace(
            go.Heatmap(
                z=z,
                x=x,
                y=y,
                colorscale=colorscale,
                zmid=0,
                colorbar={"title": "Imbalance"},
                hovertemplate="Time: %{x}<br>Symbol: %{y}<br>Imbalance: %{z:.4f}<extra></extra>",
            )
        )

        # Apply theme
        layout = self.theme.to_plotly_layout()
        layout.update(
            title=self.title,
            width=self.width,
            height=self.height,
        )
        fig.update_layout(**layout)

        return fig


class TradeImbalanceChart(BaseChart):
    """
    Trade imbalance visualization.

    Shows buy vs sell volume imbalance with cumulative tracking.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        date_col: str = "bar_start",
        buy_volume_col: str = "buy_volume",
        sell_volume_col: str = "sell_volume",
        title: str = "Trade Imbalance",
        width: int = 1200,
        height: int = 600,
        theme: ThemeConfig | None = None,
        show_cumulative: bool = True,
    ) -> None:
        """
        Initialize trade imbalance chart.

        Args:
            data: DataFrame with trade volume data.
            date_col: Column name for datetime.
            buy_volume_col: Column name for buy volume.
            sell_volume_col: Column name for sell volume.
            title: Chart title.
            width: Chart width in pixels.
            height: Chart height in pixels.
            theme: Optional theme configuration.
            show_cumulative: Whether to show cumulative imbalance.
        """
        self.date_col = date_col
        self.buy_volume_col = buy_volume_col
        self.sell_volume_col = sell_volume_col
        self.show_cumulative = show_cumulative
        super().__init__(data, title, width, height, theme)

    def _create_figure(self) -> go.Figure:
        """Create trade imbalance chart."""
        rows = 2 if self.show_cumulative else 1
        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.6, 0.4] if self.show_cumulative else [1.0],
        )

        if self.data is None:
            return fig

        has_buy = self.buy_volume_col in self.data.columns
        has_sell = self.sell_volume_col in self.data.columns

        if not has_buy and not has_sell:
            return fig

        # Calculate imbalance
        buy_vol = self.data.get(self.buy_volume_col, pd.Series(0, index=self.data.index))
        sell_vol = self.data.get(self.sell_volume_col, pd.Series(0, index=self.data.index))
        imbalance = buy_vol - sell_vol
        total_vol = buy_vol + sell_vol
        imbalance_pct = np.where(total_vol > 0, imbalance / total_vol, 0)

        # Bar colors based on direction
        colors = [
            self.theme.colors.bullish if x > 0 else self.theme.colors.bearish for x in imbalance
        ]

        # Imbalance bars
        fig.add_trace(
            go.Bar(
                x=self.data[self.date_col],
                y=imbalance,
                name="Imbalance",
                marker_color=colors,
            ),
            row=1,
            col=1,
        )

        fig.update_yaxes(title_text="Net Imbalance", row=1, col=1)

        # Cumulative imbalance line
        if self.show_cumulative:
            cumulative = imbalance.cumsum()
            fig.add_trace(
                go.Scatter(
                    x=self.data[self.date_col],
                    y=cumulative,
                    name="Cumulative",
                    mode="lines",
                    line={"color": self.theme.colors.info, "width": 2},
                    fill="tozeroy",
                    fillcolor="rgba(23, 190, 207, 0.2)",
                ),
                row=2,
                col=1,
            )
            fig.update_yaxes(title_text="Cumulative", row=2, col=1)

        # Apply theme
        layout = self.theme.to_plotly_layout()
        layout.update(
            title=self.title,
            width=self.width,
            height=self.height,
            showlegend=True,
        )
        fig.update_layout(**layout)

        return fig


def create_ohlc_chart(
    data: pd.DataFrame,
    title: str = "",
    show_volume: bool = True,
    theme: ThemeConfig | None = None,
    **kwargs: Any,
) -> CandlestickChart:
    """
    Convenience function to create a candlestick chart.

    Args:
        data: DataFrame with OHLCV data.
        title: Chart title.
        show_volume: Whether to show volume panel.
        theme: Optional theme configuration.
        **kwargs: Additional arguments passed to CandlestickChart.

    Returns:
        Configured CandlestickChart instance.
    """
    return CandlestickChart(
        data=data,
        title=title,
        show_volume=show_volume,
        theme=theme,
        **kwargs,
    )


def create_market_chart(
    data: pd.DataFrame,
    title: str = "",
    theme: ThemeConfig | None = None,
) -> MarketChart:
    """
    Convenience function to create a market chart.

    Args:
        data: DataFrame with market data.
        title: Chart title.
        theme: Optional theme configuration.

    Returns:
        Configured MarketChart instance.
    """
    return MarketChart(data=data, title=title, theme=theme)
