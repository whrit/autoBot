"""
Theme Configuration for Visualization Library.

Provides consistent styling across plotly and matplotlib charts
with support for light and dark modes.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ColorPalette:
    """Color palette for charts."""

    # Primary colors
    primary: str = "#1f77b4"
    secondary: str = "#ff7f0e"
    success: str = "#2ca02c"
    danger: str = "#d62728"
    warning: str = "#ffbb78"
    info: str = "#17becf"

    # Price colors
    bullish: str = "#26a69a"  # Green for price up
    bearish: str = "#ef5350"  # Red for price down
    neutral: str = "#78909c"  # Gray for neutral

    # Volume colors
    volume_up: str = "rgba(38, 166, 154, 0.5)"
    volume_down: str = "rgba(239, 83, 80, 0.5)"

    # Grid and background
    grid: str = "#e0e0e0"
    background: str = "#ffffff"
    paper: str = "#ffffff"
    text: str = "#212121"

    # Gradient for heatmaps
    heatmap_scale: list[str] = field(
        default_factory=lambda: [
            "#313695",
            "#4575b4",
            "#74add1",
            "#abd9e9",
            "#e0f3f8",
            "#ffffbf",
            "#fee090",
            "#fdae61",
            "#f46d43",
            "#d73027",
            "#a50026",
        ]
    )


@dataclass
class DarkColorPalette(ColorPalette):
    """Dark mode color palette."""

    grid: str = "#3a3a3a"
    background: str = "#1e1e1e"
    paper: str = "#252525"
    text: str = "#e0e0e0"


@dataclass
class ThemeConfig:
    """Complete theme configuration."""

    name: str
    colors: ColorPalette
    font_family: str = "Arial, sans-serif"
    font_size: int = 12
    title_font_size: int = 16
    axis_title_font_size: int = 14
    legend_font_size: int = 11
    line_width: float = 1.5
    marker_size: int = 6
    grid_width: float = 0.5

    def to_plotly_layout(self) -> dict[str, Any]:
        """Convert theme to plotly layout configuration."""
        return {
            "template": "plotly_dark" if "dark" in self.name.lower() else "plotly_white",
            "paper_bgcolor": self.colors.paper,
            "plot_bgcolor": self.colors.background,
            "font": {
                "family": self.font_family,
                "size": self.font_size,
                "color": self.colors.text,
            },
            "title": {
                "font": {
                    "size": self.title_font_size,
                    "color": self.colors.text,
                }
            },
            "xaxis": {
                "gridcolor": self.colors.grid,
                "gridwidth": self.grid_width,
                "zerolinecolor": self.colors.grid,
                "title": {"font": {"size": self.axis_title_font_size}},
            },
            "yaxis": {
                "gridcolor": self.colors.grid,
                "gridwidth": self.grid_width,
                "zerolinecolor": self.colors.grid,
                "title": {"font": {"size": self.axis_title_font_size}},
            },
            "legend": {
                "font": {"size": self.legend_font_size},
                "bgcolor": "rgba(0,0,0,0)",
            },
            "colorway": [
                self.colors.primary,
                self.colors.secondary,
                self.colors.success,
                self.colors.danger,
                self.colors.warning,
                self.colors.info,
            ],
        }

    def apply_to_matplotlib(self) -> None:
        """Apply theme to matplotlib defaults."""
        import matplotlib.pyplot as plt

        plt.rcParams.update(
            {
                "figure.facecolor": self.colors.paper,
                "axes.facecolor": self.colors.background,
                "axes.edgecolor": self.colors.grid,
                "axes.labelcolor": self.colors.text,
                "axes.titlesize": self.title_font_size,
                "axes.labelsize": self.axis_title_font_size,
                "axes.grid": True,
                "grid.color": self.colors.grid,
                "grid.linewidth": self.grid_width,
                "text.color": self.colors.text,
                "xtick.color": self.colors.text,
                "ytick.color": self.colors.text,
                "legend.fontsize": self.legend_font_size,
                "font.family": self.font_family.split(",")[0].strip(),
                "font.size": self.font_size,
                "lines.linewidth": self.line_width,
                "lines.markersize": self.marker_size,
            }
        )


# Pre-configured themes
LIGHT_THEME = ThemeConfig(name="light", colors=ColorPalette())
DARK_THEME = ThemeConfig(name="dark", colors=DarkColorPalette())

# Default theme
_current_theme = LIGHT_THEME


def get_theme() -> ThemeConfig:
    """Get the current active theme."""
    return _current_theme


def set_theme(theme: ThemeConfig | str) -> None:
    """
    Set the current active theme.

    Args:
        theme: Either a ThemeConfig object or 'light'/'dark' string.
    """
    global _current_theme
    if isinstance(theme, str):
        if theme.lower() == "dark":
            _current_theme = DARK_THEME
        else:
            _current_theme = LIGHT_THEME
    else:
        _current_theme = theme


def use_dark_mode(enabled: bool = True) -> None:
    """Enable or disable dark mode."""
    set_theme("dark" if enabled else "light")
