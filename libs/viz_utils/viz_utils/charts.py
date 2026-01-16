"""
Core Chart Utilities for Visualization Library.

Provides base chart classes and utilities for creating consistent
charts with plotly (interactive) and matplotlib (static).
"""

from abc import ABC, abstractmethod
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .themes import ThemeConfig, get_theme


class BaseChart(ABC):
    """Abstract base class for all chart types."""

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        title: str = "",
        width: int = 1200,
        height: int = 600,
        theme: ThemeConfig | None = None,
    ) -> None:
        """
        Initialize chart with data and configuration.

        Args:
            data: DataFrame containing chart data.
            title: Chart title.
            width: Chart width in pixels.
            height: Chart height in pixels.
            theme: Optional theme configuration (uses default if not provided).
        """
        self.data = data
        self.title = title
        self.width = width
        self.height = height
        self.theme = theme or get_theme()
        self._figure: go.Figure | None = None

    @property
    def figure(self) -> go.Figure:
        """Get the plotly figure, creating if necessary."""
        if self._figure is None:
            self._figure = self._create_figure()
        return self._figure

    @abstractmethod
    def _create_figure(self) -> go.Figure:
        """Create the plotly figure. Must be implemented by subclasses."""
        pass

    def update_layout(self, **kwargs: Any) -> "BaseChart":
        """Update the figure layout with additional parameters."""
        self.figure.update_layout(**kwargs)
        return self

    def add_annotation(
        self,
        x: Any,
        y: Any,
        text: str,
        showarrow: bool = True,
        arrowhead: int = 2,
        **kwargs: Any,
    ) -> "BaseChart":
        """Add annotation to the chart."""
        self.figure.add_annotation(
            x=x,
            y=y,
            text=text,
            showarrow=showarrow,
            arrowhead=arrowhead,
            **kwargs,
        )
        return self

    def add_hline(
        self,
        y: float,
        line_dash: str = "dash",
        line_color: str | None = None,
        annotation_text: str | None = None,
        **kwargs: Any,
    ) -> "BaseChart":
        """Add horizontal line to the chart."""
        color = line_color or self.theme.colors.neutral
        self.figure.add_hline(
            y=y,
            line_dash=line_dash,
            line_color=color,
            annotation_text=annotation_text,
            **kwargs,
        )
        return self

    def add_vline(
        self,
        x: Any,
        line_dash: str = "dash",
        line_color: str | None = None,
        annotation_text: str | None = None,
        **kwargs: Any,
    ) -> "BaseChart":
        """Add vertical line to the chart."""
        color = line_color or self.theme.colors.neutral
        self.figure.add_vline(
            x=x,
            line_dash=line_dash,
            line_color=color,
            annotation_text=annotation_text,
            **kwargs,
        )
        return self

    def add_vrect(
        self,
        x0: Any,
        x1: Any,
        fillcolor: str | None = None,
        opacity: float = 0.2,
        **kwargs: Any,
    ) -> "BaseChart":
        """Add vertical rectangle (shaded region) to the chart."""
        color = fillcolor or self.theme.colors.info
        self.figure.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=color,
            opacity=opacity,
            layer="below",
            line_width=0,
            **kwargs,
        )
        return self

    def show(self) -> None:
        """Display the chart interactively."""
        self.figure.show()

    def to_html(self, include_plotlyjs: bool | str = True) -> str:
        """Convert chart to HTML string."""
        return self.figure.to_html(include_plotlyjs=include_plotlyjs)

    def to_image(self, format: str = "png", scale: int = 2) -> bytes:
        """Convert chart to image bytes."""
        return self.figure.to_image(format=format, scale=scale)


class MultiPanelChart(BaseChart):
    """Chart with multiple vertical panels (subplots)."""

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        title: str = "",
        width: int = 1200,
        height: int = 800,
        theme: ThemeConfig | None = None,
        rows: int = 2,
        row_heights: list[float] | None = None,
        shared_xaxes: bool = True,
        vertical_spacing: float = 0.03,
    ) -> None:
        """
        Initialize multi-panel chart.

        Args:
            data: DataFrame containing chart data.
            title: Chart title.
            width: Chart width in pixels.
            height: Chart height in pixels.
            theme: Optional theme configuration.
            rows: Number of rows (panels).
            row_heights: Relative heights of rows (must sum to 1).
            shared_xaxes: Whether to share x-axes across panels.
            vertical_spacing: Spacing between panels.
        """
        self.rows = rows
        self.row_heights = row_heights or [1.0 / rows] * rows
        self.shared_xaxes = shared_xaxes
        self.vertical_spacing = vertical_spacing
        super().__init__(data, title, width, height, theme)

    def _create_figure(self) -> go.Figure:
        """Create multi-panel plotly figure."""
        fig = make_subplots(
            rows=self.rows,
            cols=1,
            shared_xaxes=self.shared_xaxes,
            vertical_spacing=self.vertical_spacing,
            row_heights=self.row_heights,
        )

        # Apply theme
        layout = self.theme.to_plotly_layout()
        layout.update(
            title=self.title,
            width=self.width,
            height=self.height,
            showlegend=True,
            legend={"x": 0, "y": 1.02, "orientation": "h"},
        )
        fig.update_layout(**layout)

        return fig

    def add_trace_to_row(
        self,
        trace: go.Scatter | go.Candlestick | go.Bar | go.Heatmap,
        row: int,
    ) -> "MultiPanelChart":
        """Add a trace to a specific row."""
        self.figure.add_trace(trace, row=row, col=1)
        return self


class SimpleLineChart(BaseChart):
    """Simple line chart for time series data."""

    def __init__(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_cols: list[str],
        title: str = "",
        width: int = 1200,
        height: int = 600,
        theme: ThemeConfig | None = None,
        fill_area: bool = False,
    ) -> None:
        """
        Initialize simple line chart.

        Args:
            data: DataFrame containing chart data.
            x_col: Column name for x-axis.
            y_cols: Column names for y-axis (multiple lines).
            title: Chart title.
            width: Chart width in pixels.
            height: Chart height in pixels.
            theme: Optional theme configuration.
            fill_area: Whether to fill area under lines.
        """
        self.x_col = x_col
        self.y_cols = y_cols
        self.fill_area = fill_area
        super().__init__(data, title, width, height, theme)

    def _create_figure(self) -> go.Figure:
        """Create simple line chart."""
        fig = go.Figure()

        colors = [
            self.theme.colors.primary,
            self.theme.colors.secondary,
            self.theme.colors.success,
            self.theme.colors.danger,
            self.theme.colors.warning,
            self.theme.colors.info,
        ]

        for i, col in enumerate(self.y_cols):
            if self.data is None:
                continue
            color = colors[i % len(colors)]
            fill = "tozeroy" if self.fill_area else None

            fig.add_trace(
                go.Scatter(
                    x=self.data[self.x_col],
                    y=self.data[col],
                    name=col,
                    mode="lines",
                    line={"color": color, "width": self.theme.line_width},
                    fill=fill,
                )
            )

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


class BarChart(BaseChart):
    """Bar chart for categorical or time series data."""

    def __init__(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str = "",
        width: int = 1200,
        height: int = 600,
        theme: ThemeConfig | None = None,
        color_col: str | None = None,
        orientation: str = "v",
    ) -> None:
        """
        Initialize bar chart.

        Args:
            data: DataFrame containing chart data.
            x_col: Column name for x-axis (or y if horizontal).
            y_col: Column name for y-axis (or x if horizontal).
            title: Chart title.
            width: Chart width in pixels.
            height: Chart height in pixels.
            theme: Optional theme configuration.
            color_col: Optional column for bar colors.
            orientation: 'v' for vertical, 'h' for horizontal.
        """
        self.x_col = x_col
        self.y_col = y_col
        self.color_col = color_col
        self.orientation = orientation
        super().__init__(data, title, width, height, theme)

    def _create_figure(self) -> go.Figure:
        """Create bar chart."""
        if self.data is None:
            return go.Figure()

        # Determine colors
        if self.color_col and self.color_col in self.data.columns:
            colors = self.data[self.color_col].apply(
                lambda x: self.theme.colors.bullish if x > 0 else self.theme.colors.bearish
            )
        else:
            colors = [self.theme.colors.primary] * len(self.data)

        fig = go.Figure()

        if self.orientation == "v":
            fig.add_trace(
                go.Bar(
                    x=self.data[self.x_col],
                    y=self.data[self.y_col],
                    marker_color=colors,
                    name=self.y_col,
                )
            )
        else:
            fig.add_trace(
                go.Bar(
                    x=self.data[self.y_col],
                    y=self.data[self.x_col],
                    marker_color=colors,
                    orientation="h",
                    name=self.y_col,
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


class HeatmapChart(BaseChart):
    """Heatmap chart for matrix data."""

    def __init__(
        self,
        data: pd.DataFrame | np.ndarray,
        title: str = "",
        width: int = 800,
        height: int = 800,
        theme: ThemeConfig | None = None,
        x_labels: list[str] | None = None,
        y_labels: list[str] | None = None,
        colorscale: str | list[str] | None = None,
        show_values: bool = True,
        value_format: str = ".2f",
    ) -> None:
        """
        Initialize heatmap chart.

        Args:
            data: DataFrame or 2D array containing heatmap values.
            title: Chart title.
            width: Chart width in pixels.
            height: Chart height in pixels.
            theme: Optional theme configuration.
            x_labels: Labels for x-axis.
            y_labels: Labels for y-axis.
            colorscale: Plotly colorscale name or list.
            show_values: Whether to show values in cells.
            value_format: Format string for values.
        """
        self.x_labels = x_labels
        self.y_labels = y_labels
        self.colorscale = colorscale
        self.show_values = show_values
        self.value_format = value_format

        # Convert DataFrame to values
        if isinstance(data, pd.DataFrame):
            self._values = data.values
            if x_labels is None:
                self.x_labels = list(data.columns)
            if y_labels is None:
                self.y_labels = list(data.index)
        else:
            self._values = data

        super().__init__(None, title, width, height, theme)

    def _create_figure(self) -> go.Figure:
        """Create heatmap chart."""
        colorscale = self.colorscale or self.theme.colors.heatmap_scale

        # Format annotation text
        text = None
        if self.show_values:
            text = [[f"{val:{self.value_format}}" for val in row] for row in self._values]

        fig = go.Figure()

        fig.add_trace(
            go.Heatmap(
                z=self._values,
                x=self.x_labels,
                y=self.y_labels,
                colorscale=colorscale,
                text=text,
                texttemplate="%{text}" if self.show_values else None,
                textfont={"size": 10},
                hovertemplate="x: %{x}<br>y: %{y}<br>value: %{z:.4f}<extra></extra>",
            )
        )

        # Apply theme
        layout = self.theme.to_plotly_layout()
        layout.update(
            title=self.title,
            width=self.width,
            height=self.height,
            xaxis={"side": "bottom"},
            yaxis={"autorange": "reversed"},  # Top to bottom for correlation-style
        )
        fig.update_layout(**layout)

        return fig


def create_matplotlib_figure(
    rows: int = 1,
    cols: int = 1,
    figsize: tuple[int, int] | None = None,
    theme: ThemeConfig | None = None,
) -> tuple[plt.Figure, np.ndarray | plt.Axes]:
    """
    Create a matplotlib figure with theme applied.

    Args:
        rows: Number of subplot rows.
        cols: Number of subplot columns.
        figsize: Figure size in inches.
        theme: Optional theme configuration.

    Returns:
        Tuple of (figure, axes array).
    """
    active_theme = theme or get_theme()
    active_theme.apply_to_matplotlib()

    if figsize is None:
        figsize = (12, 6 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.patch.set_facecolor(active_theme.colors.paper)

    return fig, axes
