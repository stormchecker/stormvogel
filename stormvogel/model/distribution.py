"""A distribution object."""

from stormvogel.model.value import Value
from dataclasses import dataclass


@dataclass(eq=False)
class Distribution[ValueType: Value, SupportType]:
    """A sparse distribution."""

    distribution: list[tuple[ValueType, SupportType]]

    @property
    def support(self) -> set[SupportType]:
        """Returns the support of this distribution."""
        return set(s for _, s in self.distribution)

    @property
    def values(self) -> list[ValueType]:
        """Returns the values of this distribution."""
        return [v for v, _ in self.distribution]

    def is_stochastic(self, precision=1e-6) -> bool:
        """Returns whether this distribution is probabilistic (i.e., sums to 1)."""
        from stormvogel.model.value import Interval
        from stormvogel.parametric import Parametric

        if any(isinstance(v, (Interval, Parametric)) for v, _ in self.distribution):
            return True

        from fractions import Fraction

        total = sum(
            float(v)
            for v, _ in self.distribution
            if isinstance(v, (int, float, Fraction))
        )
        return abs(total - 1) < precision

    def sort(self):
        """Sorts the distribution by the support's position in the model's states."""
        self.distribution.sort(key=lambda t: t[1].model.states.index(t[1]))

    def __str__(self):
        parts = []
        for value, support in self.distribution:
            parts.append(f"{value} -> {support}")
        return ", ".join(parts)

    def __add__(self, other):
        if not isinstance(other, Distribution):
            raise TypeError("Can only add Distribution to Distribution")
        combined = {}
        for value, support in self.distribution + other.distribution:
            if support in combined:
                combined[support] += value
            else:
                combined[support] = value
        return Distribution([(v, k) for k, v in combined.items()])

    def __iter__(self):
        return iter(self.distribution)

    def __len__(self) -> int:
        return len(self.distribution)
