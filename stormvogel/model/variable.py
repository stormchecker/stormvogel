"""A model variable."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Variable:
    label: str

    def __lt__(self, other):
        if not isinstance(other, Variable):
            return NotImplemented
        return str(self.label) < str(other.label)

    def __repr__(self):
        return f"Variable({self.label!r})"

    def __str__(self):
        return self.label
