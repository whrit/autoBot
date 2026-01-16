"""
Export Utilities for Visualization Library.

Provides functionality for saving charts to various formats:
- HTML (interactive)
- PNG/SVG (static)
- Multi-chart reports
- Notebook-friendly display
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

from .charts import BaseChart


def save_html(
    chart: BaseChart | go.Figure,
    path: str | Path,
    include_plotlyjs: bool | str = True,
    full_html: bool = True,
    auto_open: bool = False,
) -> Path:
    """
    Save chart to HTML file.

    Args:
        chart: Chart object or plotly Figure.
        path: Output file path.
        include_plotlyjs: Whether/how to include plotly.js.
                          True = inline, 'cdn' = CDN link, False = none.
        full_html: Whether to include full HTML document structure.
        auto_open: Whether to open in browser after saving.

    Returns:
        Path to saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = chart.figure if isinstance(chart, BaseChart) else chart

    pio.write_html(
        fig,
        str(path),
        include_plotlyjs=include_plotlyjs,
        full_html=full_html,
        auto_open=auto_open,
    )

    return path


def save_png(
    chart: BaseChart | go.Figure | plt.Figure,
    path: str | Path,
    width: int | None = None,
    height: int | None = None,
    scale: int = 2,
    dpi: int = 150,
) -> Path:
    """
    Save chart to PNG file.

    Args:
        chart: Chart object, plotly Figure, or matplotlib Figure.
        path: Output file path.
        width: Image width in pixels (plotly only).
        height: Image height in pixels (plotly only).
        scale: Scale factor for resolution (plotly only).
        dpi: DPI for matplotlib figures.

    Returns:
        Path to saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(chart, plt.Figure):
        chart.savefig(str(path), dpi=dpi, bbox_inches="tight", facecolor=chart.get_facecolor())
    else:
        fig = chart.figure if isinstance(chart, BaseChart) else chart
        pio.write_image(fig, str(path), format="png", width=width, height=height, scale=scale)

    return path


def save_svg(
    chart: BaseChart | go.Figure | plt.Figure,
    path: str | Path,
    width: int | None = None,
    height: int | None = None,
) -> Path:
    """
    Save chart to SVG file.

    Args:
        chart: Chart object, plotly Figure, or matplotlib Figure.
        path: Output file path.
        width: Image width in pixels (plotly only).
        height: Image height in pixels (plotly only).

    Returns:
        Path to saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(chart, plt.Figure):
        chart.savefig(str(path), format="svg", bbox_inches="tight", facecolor=chart.get_facecolor())
    else:
        fig = chart.figure if isinstance(chart, BaseChart) else chart
        pio.write_image(fig, str(path), format="svg", width=width, height=height)

    return path


def to_image_bytes(
    chart: BaseChart | go.Figure | plt.Figure,
    format: str = "png",
    width: int | None = None,
    height: int | None = None,
    scale: int = 2,
    dpi: int = 150,
) -> bytes:
    """
    Convert chart to image bytes.

    Args:
        chart: Chart object, plotly Figure, or matplotlib Figure.
        format: Image format ('png', 'svg', 'jpeg', 'webp', 'pdf').
        width: Image width in pixels (plotly only).
        height: Image height in pixels (plotly only).
        scale: Scale factor for resolution (plotly only).
        dpi: DPI for matplotlib figures.

    Returns:
        Image bytes.
    """
    if isinstance(chart, plt.Figure):
        buffer = BytesIO()
        chart.savefig(
            buffer, format=format, dpi=dpi, bbox_inches="tight", facecolor=chart.get_facecolor()
        )
        buffer.seek(0)
        return buffer.read()
    else:
        fig = chart.figure if isinstance(chart, BaseChart) else chart
        return fig.to_image(format=format, width=width, height=height, scale=scale)


def to_base64(
    chart: BaseChart | go.Figure | plt.Figure,
    format: str = "png",
    **kwargs: Any,
) -> str:
    """
    Convert chart to base64 encoded string.

    Useful for embedding in HTML or Markdown.

    Args:
        chart: Chart object, plotly Figure, or matplotlib Figure.
        format: Image format.
        **kwargs: Additional arguments passed to to_image_bytes.

    Returns:
        Base64 encoded string.
    """
    image_bytes = to_image_bytes(chart, format=format, **kwargs)
    return base64.b64encode(image_bytes).decode("utf-8")


def to_data_uri(
    chart: BaseChart | go.Figure | plt.Figure,
    format: str = "png",
    **kwargs: Any,
) -> str:
    """
    Convert chart to data URI for embedding in HTML.

    Args:
        chart: Chart object, plotly Figure, or matplotlib Figure.
        format: Image format.
        **kwargs: Additional arguments passed to to_image_bytes.

    Returns:
        Data URI string.
    """
    mime_types = {
        "png": "image/png",
        "jpeg": "image/jpeg",
        "jpg": "image/jpeg",
        "svg": "image/svg+xml",
        "webp": "image/webp",
        "pdf": "application/pdf",
    }
    mime_type = mime_types.get(format.lower(), "image/png")
    b64 = to_base64(chart, format=format, **kwargs)
    return f"data:{mime_type};base64,{b64}"


def display_inline(
    chart: BaseChart | go.Figure | plt.Figure,
) -> None:
    """
    Display chart inline (for notebooks).

    Args:
        chart: Chart object, plotly Figure, or matplotlib Figure.
    """
    if isinstance(chart, plt.Figure):
        plt.show()
    else:
        fig = chart.figure if isinstance(chart, BaseChart) else chart
        fig.show()


class MultiChartReport:
    """
    Multi-chart report builder.

    Combines multiple charts into a single HTML report with
    sections and descriptions.
    """

    def __init__(
        self,
        title: str = "Chart Report",
        description: str = "",
    ) -> None:
        """
        Initialize report builder.

        Args:
            title: Report title.
            description: Report description/summary.
        """
        self.title = title
        self.description = description
        self.sections: list[dict[str, Any]] = []

    def add_chart(
        self,
        chart: BaseChart | go.Figure,
        title: str = "",
        description: str = "",
    ) -> "MultiChartReport":
        """
        Add a chart to the report.

        Args:
            chart: Chart to add.
            title: Section title.
            description: Section description.

        Returns:
            Self for chaining.
        """
        fig = chart.figure if isinstance(chart, BaseChart) else chart
        self.sections.append(
            {
                "type": "chart",
                "figure": fig,
                "title": title,
                "description": description,
            }
        )
        return self

    def add_text(
        self,
        text: str,
        title: str = "",
    ) -> "MultiChartReport":
        """
        Add a text section to the report.

        Args:
            text: Text content (supports HTML).
            title: Section title.

        Returns:
            Self for chaining.
        """
        self.sections.append(
            {
                "type": "text",
                "content": text,
                "title": title,
            }
        )
        return self

    def add_table(
        self,
        data: Any,
        title: str = "",
        description: str = "",
    ) -> "MultiChartReport":
        """
        Add a data table to the report.

        Args:
            data: DataFrame or dict to display as table.
            title: Section title.
            description: Section description.

        Returns:
            Self for chaining.
        """
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            html_table = data.to_html(classes=["data-table"], border=0)
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
            html_table = df.to_html(classes=["data-table"], border=0)
        else:
            html_table = str(data)

        self.sections.append(
            {
                "type": "table",
                "content": html_table,
                "title": title,
                "description": description,
            }
        )
        return self

    def _generate_html(self, include_plotlyjs: bool | str = True) -> str:
        """Generate the full HTML report."""
        # CSS styles
        css = """
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #333;
                border-bottom: 2px solid #1f77b4;
                padding-bottom: 10px;
            }
            h2 {
                color: #555;
                margin-top: 30px;
            }
            .report-description {
                color: #666;
                font-size: 1.1em;
                margin-bottom: 30px;
            }
            .section {
                background: white;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .section-description {
                color: #666;
                margin-bottom: 15px;
            }
            .chart-container {
                width: 100%;
                overflow-x: auto;
            }
            .data-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.9em;
            }
            .data-table th, .data-table td {
                padding: 8px 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            .data-table th {
                background-color: #f8f8f8;
                font-weight: 600;
            }
            .data-table tr:hover {
                background-color: #f5f5f5;
            }
            .text-content {
                line-height: 1.6;
            }
        </style>
        """

        # Plotly.js include
        if include_plotlyjs is True:
            plotly_js = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
        elif include_plotlyjs == "cdn":
            plotly_js = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
        else:
            plotly_js = ""

        # Build sections HTML
        sections_html = []
        for i, section in enumerate(self.sections):
            section_html = '<div class="section">'

            if section.get("title"):
                section_html += f'<h2>{section["title"]}</h2>'

            if section.get("description"):
                section_html += f'<p class="section-description">{section["description"]}</p>'

            if section["type"] == "chart":
                fig = section["figure"]
                chart_html = pio.to_html(fig, full_html=False, include_plotlyjs=False)
                section_html += f'<div class="chart-container">{chart_html}</div>'

            elif section["type"] == "text":
                section_html += f'<div class="text-content">{section["content"]}</div>'

            elif section["type"] == "table":
                section_html += f'<div class="table-container">{section["content"]}</div>'

            section_html += "</div>"
            sections_html.append(section_html)

        # Build full HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    {plotly_js}
    {css}
</head>
<body>
    <h1>{self.title}</h1>
    <p class="report-description">{self.description}</p>
    {"".join(sections_html)}
</body>
</html>
"""
        return html

    def save(
        self,
        path: str | Path,
        include_plotlyjs: bool | str = True,
    ) -> Path:
        """
        Save report to HTML file.

        Args:
            path: Output file path.
            include_plotlyjs: Whether/how to include plotly.js.

        Returns:
            Path to saved file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        html = self._generate_html(include_plotlyjs=include_plotlyjs)

        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

        return path

    def to_html(self, include_plotlyjs: bool | str = True) -> str:
        """
        Generate HTML string.

        Args:
            include_plotlyjs: Whether/how to include plotly.js.

        Returns:
            HTML string.
        """
        return self._generate_html(include_plotlyjs=include_plotlyjs)


def create_report(
    title: str = "Chart Report",
    description: str = "",
) -> MultiChartReport:
    """
    Create a new multi-chart report.

    Args:
        title: Report title.
        description: Report description.

    Returns:
        MultiChartReport instance.
    """
    return MultiChartReport(title=title, description=description)
