"""A distribution object."""

from stormvogel.model.value import Value
from dataclasses import dataclass


@dataclass
class Distribution[ValueType: Value, SupportType]:
    """Represent a sparse distribution mapping values to support elements."""

    _distribution: list[tuple[ValueType, SupportType]]

    @property
    def support(self) -> set[SupportType]:
        """Return the support of this distribution."""
        return set(s for prob, s in self._distribution if not prob.is_zero())

    @property
    def probabilities(self) -> list[ValueType]:
        """Returns the values of this distribution."""
        return [v for v, _ in self._distribution]

    def is_stochastic(self, epsilon=1e-6) -> bool:
        """Check whether this distribution is probabilistic (i.e., sums to 1).

        :param precision: Tolerance for floating-point comparison.
        """
        from stormvogel.model.value import Interval
        from stormvogel.parametric import Parametric

        if any(isinstance(v, (Interval, Parametric)) for v, _ in self._distribution):
            return True

        from fractions import Fraction

        total = sum(
            float(v)
            for v, _ in self._distribution
            if isinstance(v, (int, float, Fraction))
        )
        return abs(total - 1) < epsilon

    def __str__(self):
        parts = []
        for value, support in self._distribution:
            parts.append(f"{value} -> {support}")
        return ", ".join(parts)

    def __add__(self, other):
        if not isinstance(other, Distribution):
            raise TypeError("Can only add Distribution to Distribution")
        combined = {}
        for value, support in self._distribution + other._distribution:
            if support in combined:
                combined[support] += value
            else:
                combined[support] = value
        return Distribution([(v, k) for k, v in combined.items()])

    def __iter__(self):
        return iter(self._distribution)

    def __len__(self) -> int:
        return len(self._distribution)
