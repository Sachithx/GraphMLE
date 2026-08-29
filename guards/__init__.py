"""Pre-compute safety guards for candidate pipelines."""

from .leakage import LeakageError, LeakageGuard

__all__ = ["LeakageError", "LeakageGuard"]
