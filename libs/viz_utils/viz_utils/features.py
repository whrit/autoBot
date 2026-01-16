"""
Feature Visualization Module.

Provides charts for analyzing ML features including:
- Feature distribution histograms
- Feature correlation heatmaps
- Time series of features
- Rolling statistics plots
- Feature importance bar charts
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

from .charts import BaseChart, HeatmapChart, create_matplotlib_figure
from .themes import ThemeConfig, get_theme


class FeatureDistributionChart(BaseChart):
    """
    Feature distribution histograms.

    Visualizes the distribution of one or more features with
    optional statistical annotations.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: list[str],
        title: str = "Feature Distributions",
        width: int = 1200,
        height: int = 600,
        theme: ThemeConfig | None = None,
        bins: int = 50,
        show_kde: bool = True,
        show_stats: bool = True,
    ) -> None:
        """
        Initialize feature distribution chart.

        Args:
            data: DataFrame with feature data.
            feature_cols: List of feature column names.
            title: Chart title.
            width: Chart width in pixels.
            height: Chart height in pixels.
            theme: Optional theme configuration.
            bins: Number of histogram bins.
            show_kde: Whether to show KDE overlay.
            show_stats: Whether to show mean/std annotations.
        """
        self.feature_cols = feature_cols
        self.bins = bins
        self.show_kde = show_kde
        self.show_stats = show_stats
        super().__init__(data, title, width, height, theme)

    def _create_figure(self) -> go.Figure:
        """Create feature distribution chart."""
        n_features = len(self.feature_cols)
        cols = min(3, n_features)
        rows = (n_features + cols - 1) // cols

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=self.feature_cols,
            vertical_spacing=0.1,
            horizontal_spacing=0.08,
        )

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

        for i, col in enumerate(self.feature_cols):
            if col not in self.data.columns:
                continue

            row = i // cols + 1
            col_idx = i % cols + 1
            color = colors[i % len(colors)]

            values = self.data[col].dropna()

            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=values,
                    nbinsx=self.bins,
                    name=col,
                    marker_color=color,
                    opacity=0.7,
                    showlegend=False,
                ),
                row=row,
                col=col_idx,
            )

            # Add stats annotation
            if self.show_stats:
                mean_val = values.mean()
                std_val = values.std()
                fig.add_annotation(
                    x=0.95,
                    y=0.95,
                    xref=f"x{i + 1 if i > 0 else ''} domain",
                    yref=f"y{i + 1 if i > 0 else ''} domain",
                    text=f"μ={mean_val:.3f}<br>σ={std_val:.3f}",
                    showarrow=False,
                    font={"size": 10},
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor=color,
                    borderwidth=1,
                )

        # Apply theme
        layout = self.theme.to_plotly_layout()
        layout.update(
            title=self.title,
            width=self.width,
            height=max(300, 250 * rows),
            showlegend=False,
        )
        fig.update_layout(**layout)

        return fig


class FeatureCorrelationHeatmap(HeatmapChart):
    """
    Feature correlation heatmap.

    Visualizes pairwise correlations between features with
    hierarchical clustering option.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: list[str] | None = None,
        title: str = "Feature Correlations",
        width: int = 900,
        height: int = 900,
        theme: ThemeConfig | None = None,
        method: str = "pearson",
        cluster: bool = False,
    ) -> None:
        """
        Initialize feature correlation heatmap.

        Args:
            data: DataFrame with feature data.
            feature_cols: List of feature columns (all numeric if None).
            title: Chart title.
            width: Chart width in pixels.
            height: Chart height in pixels.
            theme: Optional theme configuration.
            method: Correlation method ('pearson', 'spearman', 'kendall').
            cluster: Whether to cluster features.
        """
        self.method = method
        self.cluster = cluster

        # Select numeric columns if not specified
        if feature_cols is None:
            feature_cols = list(data.select_dtypes(include=[np.number]).columns)

        self.feature_cols = feature_cols

        # Calculate correlation matrix
        subset = data[feature_cols].dropna()
        corr_matrix = subset.corr(method=method)

        # Cluster if requested
        if cluster and len(corr_matrix) > 2:
            from scipy.cluster.hierarchy import leaves_list, linkage

            linkage_matrix = linkage(corr_matrix.values, method="average")
            order = leaves_list(linkage_matrix)
            corr_matrix = corr_matrix.iloc[order, order]

        super().__init__(
            data=corr_matrix,
            title=title,
            width=width,
            height=height,
            theme=theme,
            colorscale="RdBu_r",
            show_values=True,
            value_format=".2f",
        )


class FeatureTimeSeriesChart(BaseChart):
    """
    Feature time series visualization.

    Shows feature evolution over time with optional
    normalization and overlays.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        date_col: str,
        feature_cols: list[str],
        title: str = "Feature Time Series",
        width: int = 1200,
        height: int = 600,
        theme: ThemeConfig | None = None,
        normalize: bool = False,
        stacked: bool = False,
    ) -> None:
        """
        Initialize feature time series chart.

        Args:
            data: DataFrame with feature data.
            date_col: Column name for datetime.
            feature_cols: List of feature column names.
            title: Chart title.
            width: Chart width in pixels.
            height: Chart height in pixels.
            theme: Optional theme configuration.
            normalize: Whether to z-score normalize features.
            stacked: Whether to create stacked subplots.
        """
        self.date_col = date_col
        self.feature_cols = feature_cols
        self.normalize = normalize
        self.stacked = stacked
        super().__init__(data, title, width, height, theme)

    def _create_figure(self) -> go.Figure:
        """Create feature time series chart."""
        if self.stacked:
            n_features = len(self.feature_cols)
            fig = make_subplots(
                rows=n_features,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                subplot_titles=self.feature_cols,
            )
        else:
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

        for i, col in enumerate(self.feature_cols):
            if col not in self.data.columns:
                continue

            values = self.data[col].copy()

            # Normalize if requested
            if self.normalize:
                mean = values.mean()
                std = values.std()
                values = (values - mean) / std if std > 0 else values - mean

            color = colors[i % len(colors)]

            trace = go.Scatter(
                x=self.data[self.date_col],
                y=values,
                name=col,
                mode="lines",
                line={"color": color, "width": self.theme.line_width},
            )

            if self.stacked:
                fig.add_trace(trace, row=i + 1, col=1)
                fig.update_yaxes(title_text=col, row=i + 1, col=1)
            else:
                fig.add_trace(trace)

        # Apply theme
        layout = self.theme.to_plotly_layout()
        y_title = "Z-Score" if self.normalize else "Value"
        height = max(400, 150 * len(self.feature_cols)) if self.stacked else self.height
        layout.update(
            title=self.title,
            width=self.width,
            height=height,
            showlegend=not self.stacked,
        )

        if not self.stacked:
            layout.update(yaxis_title=y_title)

        fig.update_layout(**layout)

        return fig


class RollingStatisticsChart(BaseChart):
    """
    Rolling statistics visualization.

    Shows rolling mean, std, and percentile bands for features.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        date_col: str,
        value_col: str,
        title: str = "Rolling Statistics",
        width: int = 1200,
        height: int = 600,
        theme: ThemeConfig | None = None,
        window: int = 20,
        show_percentiles: bool = True,
        percentiles: tuple[float, float] = (0.05, 0.95),
    ) -> None:
        """
        Initialize rolling statistics chart.

        Args:
            data: DataFrame with feature data.
            date_col: Column name for datetime.
            value_col: Column name for values.
            title: Chart title.
            width: Chart width in pixels.
            height: Chart height in pixels.
            theme: Optional theme configuration.
            window: Rolling window size.
            show_percentiles: Whether to show percentile bands.
            percentiles: Lower and upper percentile values.
        """
        self.date_col = date_col
        self.value_col = value_col
        self.window = window
        self.show_percentiles = show_percentiles
        self.percentiles = percentiles
        super().__init__(data, title, width, height, theme)

    def _create_figure(self) -> go.Figure:
        """Create rolling statistics chart."""
        fig = go.Figure()

        if self.data is None or self.value_col not in self.data.columns:
            return fig

        values = self.data[self.value_col]
        dates = self.data[self.date_col]

        # Raw values
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                name="Value",
                mode="lines",
                line={"color": self.theme.colors.neutral, "width": 0.5},
                opacity=0.5,
            )
        )

        # Rolling mean
        rolling_mean = values.rolling(window=self.window, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=rolling_mean,
                name=f"MA({self.window})",
                mode="lines",
                line={"color": self.theme.colors.primary, "width": 2},
            )
        )

        # Rolling std bands
        rolling_std = values.rolling(window=self.window, min_periods=1).std()
        upper = rolling_mean + rolling_std
        lower = rolling_mean - rolling_std

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=upper,
                name="+1σ",
                mode="lines",
                line={"color": self.theme.colors.secondary, "width": 1, "dash": "dash"},
            )
        )

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=lower,
                name="-1σ",
                mode="lines",
                line={"color": self.theme.colors.secondary, "width": 1, "dash": "dash"},
                fill="tonexty",
                fillcolor="rgba(255, 127, 14, 0.1)",
            )
        )

        # Percentile bands
        if self.show_percentiles:
            lower_pct = values.rolling(window=self.window, min_periods=1).quantile(
                self.percentiles[0]
            )
            upper_pct = values.rolling(window=self.window, min_periods=1).quantile(
                self.percentiles[1]
            )

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=upper_pct,
                    name=f"P{int(self.percentiles[1] * 100)}",
                    mode="lines",
                    line={"color": self.theme.colors.info, "width": 1, "dash": "dot"},
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=lower_pct,
                    name=f"P{int(self.percentiles[0] * 100)}",
                    mode="lines",
                    line={"color": self.theme.colors.info, "width": 1, "dash": "dot"},
                )
            )

        # Apply theme
        layout = self.theme.to_plotly_layout()
        layout.update(
            title=self.title,
            width=self.width,
            height=self.height,
            showlegend=True,
            yaxis_title=self.value_col,
        )
        fig.update_layout(**layout)

        return fig


class FeatureImportanceChart(BaseChart):
    """
    Feature importance bar chart.

    Visualizes feature importance scores with optional
    confidence intervals.
    """

    def __init__(
        self,
        feature_names: list[str],
        importance_values: list[float] | np.ndarray,
        title: str = "Feature Importance",
        width: int = 900,
        height: int = 600,
        theme: ThemeConfig | None = None,
        errors: list[float] | np.ndarray | None = None,
        top_n: int | None = None,
        horizontal: bool = True,
        sort: bool = True,
    ) -> None:
        """
        Initialize feature importance chart.

        Args:
            feature_names: List of feature names.
            importance_values: List of importance values.
            title: Chart title.
            width: Chart width in pixels.
            height: Chart height in pixels.
            theme: Optional theme configuration.
            errors: Optional error bars for importance values.
            top_n: Show only top N features.
            horizontal: Whether to use horizontal bars.
            sort: Whether to sort by importance.
        """
        self.feature_names = list(feature_names)
        self.importance_values = np.array(importance_values)
        self.errors = np.array(errors) if errors is not None else None
        self.top_n = top_n
        self.horizontal = horizontal
        self.sort = sort
        super().__init__(None, title, width, height, theme)

    def _create_figure(self) -> go.Figure:
        """Create feature importance chart."""
        fig = go.Figure()

        names = self.feature_names.copy()
        values = self.importance_values.copy()
        errors = self.errors.copy() if self.errors is not None else None

        # Sort if requested
        if self.sort:
            idx = np.argsort(values)[::-1]
            names = [names[i] for i in idx]
            values = values[idx]
            if errors is not None:
                errors = errors[idx]

        # Limit to top N
        if self.top_n and self.top_n < len(names):
            names = names[: self.top_n]
            values = values[: self.top_n]
            if errors is not None:
                errors = errors[: self.top_n]

        # Reverse for horizontal (so highest at top)
        if self.horizontal:
            names = names[::-1]
            values = values[::-1]
            if errors is not None:
                errors = errors[::-1]

        # Color gradient based on importance
        max_val = max(values) if len(values) > 0 else 1
        colors = [
            f"rgba(31, 119, 180, {0.3 + 0.7 * (v / max_val)})" for v in values
        ]

        error_y = None
        error_x = None
        if errors is not None:
            if self.horizontal:
                error_x = {"type": "data", "array": errors, "visible": True}
            else:
                error_y = {"type": "data", "array": errors, "visible": True}

        if self.horizontal:
            fig.add_trace(
                go.Bar(
                    x=values,
                    y=names,
                    orientation="h",
                    marker_color=colors,
                    error_x=error_x,
                    name="Importance",
                )
            )
        else:
            fig.add_trace(
                go.Bar(
                    x=names,
                    y=values,
                    marker_color=colors,
                    error_y=error_y,
                    name="Importance",
                )
            )

        # Apply theme
        layout = self.theme.to_plotly_layout()
        layout.update(
            title=self.title,
            width=self.width,
            height=max(400, len(names) * 25) if self.horizontal else self.height,
            showlegend=False,
        )

        if self.horizontal:
            layout.update(xaxis_title="Importance")
        else:
            layout.update(yaxis_title="Importance", xaxis_tickangle=-45)

        fig.update_layout(**layout)

        return fig


def create_feature_distribution(
    data: pd.DataFrame,
    feature_cols: list[str],
    title: str = "Feature Distributions",
    theme: ThemeConfig | None = None,
    **kwargs: Any,
) -> FeatureDistributionChart:
    """
    Convenience function to create feature distribution chart.

    Args:
        data: DataFrame with feature data.
        feature_cols: List of feature column names.
        title: Chart title.
        theme: Optional theme configuration.
        **kwargs: Additional arguments.

    Returns:
        Configured FeatureDistributionChart instance.
    """
    return FeatureDistributionChart(
        data=data, feature_cols=feature_cols, title=title, theme=theme, **kwargs
    )


def create_correlation_heatmap(
    data: pd.DataFrame,
    feature_cols: list[str] | None = None,
    title: str = "Feature Correlations",
    theme: ThemeConfig | None = None,
    **kwargs: Any,
) -> FeatureCorrelationHeatmap:
    """
    Convenience function to create correlation heatmap.

    Args:
        data: DataFrame with feature data.
        feature_cols: List of feature columns.
        title: Chart title.
        theme: Optional theme configuration.
        **kwargs: Additional arguments.

    Returns:
        Configured FeatureCorrelationHeatmap instance.
    """
    return FeatureCorrelationHeatmap(
        data=data, feature_cols=feature_cols, title=title, theme=theme, **kwargs
    )


def create_feature_importance(
    feature_names: list[str],
    importance_values: list[float] | np.ndarray,
    title: str = "Feature Importance",
    theme: ThemeConfig | None = None,
    **kwargs: Any,
) -> FeatureImportanceChart:
    """
    Convenience function to create feature importance chart.

    Args:
        feature_names: List of feature names.
        importance_values: List of importance values.
        title: Chart title.
        theme: Optional theme configuration.
        **kwargs: Additional arguments.

    Returns:
        Configured FeatureImportanceChart instance.
    """
    return FeatureImportanceChart(
        feature_names=feature_names,
        importance_values=importance_values,
        title=title,
        theme=theme,
        **kwargs,
    )


def plot_feature_distributions_matplotlib(
    data: pd.DataFrame,
    feature_cols: list[str],
    figsize: tuple[int, int] | None = None,
    theme: ThemeConfig | None = None,
) -> plt.Figure:
    """
    Create feature distribution plot using matplotlib.

    Args:
        data: DataFrame with feature data.
        feature_cols: List of feature column names.
        figsize: Figure size in inches.
        theme: Optional theme configuration.

    Returns:
        Matplotlib Figure.
    """
    n_features = len(feature_cols)
    cols = min(3, n_features)
    rows = (n_features + cols - 1) // cols

    if figsize is None:
        figsize = (4 * cols, 3 * rows)

    fig, axes = create_matplotlib_figure(rows=rows, cols=cols, figsize=figsize, theme=theme)

    # Flatten axes for easy iteration
    if rows == 1 and cols == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, col in enumerate(feature_cols):
        if i >= len(axes_flat):
            break
        if col not in data.columns:
            continue

        ax = axes_flat[i]
        values = data[col].dropna()

        ax.hist(values, bins=50, alpha=0.7, edgecolor="white")
        ax.axvline(values.mean(), color="red", linestyle="--", label=f"Mean: {values.mean():.3f}")
        ax.set_title(col)
        ax.legend(fontsize=8)

    # Hide unused axes
    for i in range(n_features, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.tight_layout()
    return fig


def plot_correlation_matrix_matplotlib(
    data: pd.DataFrame,
    feature_cols: list[str] | None = None,
    figsize: tuple[int, int] = (10, 10),
    theme: ThemeConfig | None = None,
    annot: bool = True,
) -> plt.Figure:
    """
    Create correlation matrix heatmap using matplotlib/seaborn.

    Args:
        data: DataFrame with feature data.
        feature_cols: List of feature columns.
        figsize: Figure size in inches.
        theme: Optional theme configuration.
        annot: Whether to annotate cells with values.

    Returns:
        Matplotlib Figure.
    """
    active_theme = theme or get_theme()
    active_theme.apply_to_matplotlib()

    if feature_cols is None:
        feature_cols = list(data.select_dtypes(include=[np.number]).columns)

    corr = data[feature_cols].corr()

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(active_theme.colors.paper)

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=annot,
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        fmt=".2f",
        annot_kws={"size": 8},
    )

    ax.set_title("Feature Correlation Matrix", fontsize=14)
    fig.tight_layout()

    return fig
