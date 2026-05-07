"""Pipeline context state for decorator integration.

Provides set_pipeline_context / clear_pipeline_context used by the
progress_task decorator in decorators.py to access the active ProgressManager.

Copyright (C) 2025 Matti Kaupenjohann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

from __future__ import annotations

from typing import Any

# Global context that pipeline can set
_pipeline_context: dict[str, Any] | None = None


def set_pipeline_context(context: dict[str, Any]) -> None:
    """Set pipeline context (called by PipelineExecutor).

    Args:
        context: Dictionary with permanences and controller reference.
    """
    global _pipeline_context
    _pipeline_context = context


def clear_pipeline_context() -> None:
    """Clear pipeline context."""
    global _pipeline_context
    _pipeline_context = None


__all__ = [
    "clear_pipeline_context",
    "set_pipeline_context",
]
