from dataclasses import dataclass
from fractions import Fraction

Number = int | float | Fraction


@dataclass
class Polynomial:
    """Represent a polynomial, to be used as a value for parametric models.

    Polynomials are represented as a dictionary with n-dimensional tuples as keys.

    :param terms: Terms of the polynomial (dictionary that relates exponents to coefficients).
    :param variables: Variables of the polynomial as a list of strings.
    """

    terms: dict[tuple, float]
    variables: list[str]

    def __init__(self, variables: list[str]):
        self.terms = dict()
        self.variables = variables

    def is_zero(self) -> bool:
        """Returns whether this polynomial is the zero polynomial."""
        return all(coefficient == 0 for coefficient in self.terms.values())

    # TODO exponents may also be a single integer
    def add_term(self, exponents: tuple[int, ...], coefficient: float):
        """Add a term to the polynomial.

        Example: ``add_term((1,2,3,4), 5)`` means we add ``5*(x1^1*x2^2*x3^3*x4^4)``.

        :param exponents: Tuple of exponents for each variable.
        :param coefficient: Coefficient of the term.
        :raises RuntimeError: If a term with the given exponents already exists,
            or if the exponents tuple has the wrong dimension.
        """

        assert isinstance(exponents, tuple)

        if exponents in self.terms.keys():
            raise RuntimeError(
                "There is already a term with these exponents in this polynomial"
            )

        if self.terms != {}:
            my_dimension = self.get_dimension()
            term_dimension = len(list(exponents))

            if my_dimension != term_dimension:
                raise RuntimeError(
                    f"The length of the exponents tuple should be: {my_dimension}"
                )
        self.terms[exponents] = float(coefficient)

    def get_dimension(self) -> int:
        """Return the number of different variables present."""
        return len(self.variables)

    def get_variables(self) -> set[str]:
        """Return the set of parameters."""
        return set(self.variables)

    def get_degree(self) -> int | None:
        """Return the degree of the polynomial.

        :returns: The degree of the polynomial.
        :raises RuntimeError: If the polynomial has no terms.
        """
        if self.terms is not {}:
            largest = 0
            for term in self.terms.keys():
                current = sum(list(term))
                if current > largest:
                    largest = current
            return largest
        raise RuntimeError("A polynomial without terms does not have a degree.")

    def evaluate(self, values: dict[str, Number]) -> float:
        """Evaluate the polynomial with the given values for the variables.

        :param values: Mapping from variable names to their values.
        :returns: The result of evaluating the polynomial.
        """
        result = 0
        for exponents, coefficient in self.terms.items():
            term = coefficient
            for variable, exponent in enumerate(exponents):
                term *= values[self.variables[variable]] ** exponent
            result += term
        return result

    def __str__(self) -> str:
        s = ""
        # we iterate through each term
        for exponents, coefficient in self.terms.items():
            # we only print terms with nonzero coefficients
            if coefficient != 0:
                # we don't print coefficients that are 1
                if coefficient != 1:
                    s += f"{coefficient}*"

                # we print the variables with their corresponding powers
                # if the tuple only consists of zeroes then we are left with 1
                all_zero = True
                for variable, exponent in enumerate(exponents):
                    if exponent != 0:
                        all_zero = False
                        s += f"{self.variables[variable]}"
                        if exponent != 1:
                            s += f"^{exponent}"
                if all_zero:
                    s += "1"
                s += " + "

        return s[:-3]

    def __lt__(self, other) -> bool:
        # we first compare by degree
        self_deg = self.get_degree()
        other_deg = other.get_degree()
        if self_deg != other_deg:
            return self_deg < other_deg

        # if the degrees are equal we compare the terms lexicografically
        return sorted(self.terms.items()) < sorted(other.terms.items())

    def __eq__(self, other) -> bool:
        if not isinstance(other, Polynomial):
            return False

        return self.terms == other.terms

    def __iter__(self):
        return iter(self.terms.items())


class RationalFunction:
    """Represent a rational function, to be used as a value for parametric models.

    Rational functions are represented as a pair of polynomials.

    :param numerator: Polynomial in the numerator.
    :param denominator: Polynomial in the denominator.
    """

    numerator: Polynomial
    denominator: Polynomial

    def __init__(self, numerator: Polynomial, denominator: Polynomial):
        denominator_all_zero = True
        for exponents, coefficient in denominator.terms.items():
            if coefficient != 0:
                denominator_all_zero = False

        if not denominator_all_zero:
            self.numerator = numerator
            self.denominator = denominator
        else:
            raise RuntimeError("dividision by 0 is not allowed")

    def is_zero(self) -> bool:
        """Returns whether this rational function is the zero function."""
        return self.numerator.is_zero()

    def get_dimension(self) -> int:
        """Return the number of different variables present."""
        return max(self.numerator.get_dimension(), self.denominator.get_dimension())

    def get_variables(self) -> set[str]:
        """Return the total set of variables of this rational function."""
        return set(self.numerator.variables).union(set(self.denominator.variables))

    def evaluate(self, values: dict[str, Number]) -> float:
        """Evaluate the rational function with the given values.

        :param values: Mapping from variable names to their values.
        :returns: The result of evaluating the rational function.
        """
        return self.numerator.evaluate(values) / self.denominator.evaluate(values)

    def __str__(self) -> str:
        s = "(" + str(self.numerator) + ")/(" + str(self.denominator) + ")"
        return s

    def __lt__(self, other) -> bool:
        if not isinstance(other, RationalFunction):
            return NotImplemented

        if self.numerator != other.numerator:
            return self.numerator < other.numerator
        return self.denominator < other.denominator


Parametric = Polynomial | RationalFunction
