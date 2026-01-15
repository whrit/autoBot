"""
Pytest configuration for backtester_py integration tests.

This module configures sys.path to include workspace packages that
aren't installed in the virtual environment.
"""

import sys
from pathlib import Path


def pytest_configure(config: object) -> None:
    """Add workspace packages to sys.path before test collection."""
    project_root = Path(__file__).parent.parent.parent.parent

    # Add workspace package directories to sys.path
    workspace_packages = [
        project_root / "libs" / "cost_models",
        project_root / "libs" / "risk_models",
        project_root / "libs" / "common_types",
        project_root / "services" / "feature_builder_py",
        project_root / "services" / "labeler_py",
        project_root / "services" / "backtester_py",
    ]

    for pkg_path in workspace_packages:
        pkg_str = str(pkg_path)
        if pkg_str not in sys.path:
            sys.path.insert(0, pkg_str)
