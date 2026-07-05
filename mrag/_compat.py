"""Compatibility helpers for optional third-party dependencies."""

from __future__ import annotations

import math
from typing import Iterable, List

try:
    import numpy as np  # type: ignore
except ImportError:
    class _Vector(list):
        """Minimal list-backed stand-in for the numpy array API used by mRAG."""

        @property
        def shape(self) -> tuple[int]:
            return (len(self),)

        def tolist(self) -> List[float]:
            return list(self)

    class _Linalg:
        @staticmethod
        def norm(values: Iterable[float]) -> float:
            return math.sqrt(sum(float(v) * float(v) for v in values))

    class _FallbackNumpy:
        ndarray = _Vector
        float32 = float
        linalg = _Linalg()

        @staticmethod
        def array(values: Iterable[float], dtype=None) -> _Vector:
            del dtype
            return _Vector(float(v) for v in values)

        @staticmethod
        def zeros(size: int, dtype=None) -> _Vector:
            del dtype
            return _Vector([0.0] * size)

        @staticmethod
        def ones(size: int, dtype=None) -> _Vector:
            del dtype
            return _Vector([1.0] * size)

        @staticmethod
        def dot(a: Iterable[float], b: Iterable[float]) -> float:
            return sum(float(x) * float(y) for x, y in zip(a, b))

    np = _FallbackNumpy()
