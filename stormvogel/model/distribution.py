"""A distribution object."""

import math

from stormvogel.model.value import Value, is_zero


class Distribution[ValueType: Value, SupportType]:
    """A sparse distribution mapping support elements to probability values."""

    _distribution: dict[SupportType, ValueType]

    def __init__(self, distribution: dict[SupportType, ValueType]):
        self._distribution = distribution

    @property
    def support(self) -> set[SupportType]:
        """Returns the support of this distribution (elements with non-zero probability)."""
        return {s for s, v in self._distribution.items() if not is_zero(v)}

    @property
    def probabilities(self) -> list[ValueType]:
        """Returns the probability values of this distribution."""
        return list(self._distribution.values())

    def is_stochastic(self, epsilon: float = 1e-6) -> bool:
        """Returns whether this distribution sums to 1."""
        from stormvogel.model.value import Interval
        from stormvogel.parametric import Parametric

        if any(
            isinstance(v, (Interval, Parametric)) for v in self._distribution.values()
        ):
            return True

        from fractions import Fraction

        total = sum(
            float(v)
            for v in self._distribution.values()
            if isinstance(v, (int, float, Fraction))
        )
        return math.isclose(total, 1, abs_tol=epsilon)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Distribution):
            return NotImplemented
        return self._distribution == other._distribution

    def __str__(self) -> str:
        parts = [f"{v} -> {s}" for s, v in self._distribution.items()]
        return ", ".join(parts)

    def __add__(
        self, other: "Distribution[ValueType, SupportType]"
    ) -> "Distribution[ValueType, SupportType]":
        if not isinstance(other, Distribution):
            raise TypeError("Can only add Distribution to Distribution")
        combined: dict[SupportType, ValueType] = dict(self._distribution)
        for support, value in other._distribution.items():
            if support in combined:
                combined[support] += value  # type: ignore[assignment]
            else:
                combined[support] = value
        return Distribution(combined)

    def __iter__(self):
        return iter((v, s) for s, v in self._distribution.items())

    def __len__(self) -> int:
        return len(self._distribution)

    def __getitem__(self, key: SupportType) -> ValueType:
        return self._distribution[key]

    def __contains__(self, key: SupportType) -> bool:
        return key in self._distribution
