"""Exception types for the PLN package."""

from __future__ import annotations


class PLNInputError(ValueError):
    """Raised when inputs violate the documented contract (shapes, ranges, duplicates, label values)."""


class PLNConfigError(ValueError):
    """Raised when a configuration is inconsistent."""
