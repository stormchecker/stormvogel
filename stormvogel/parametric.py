from dataclasses import dataclass
from fractions import Fraction

Number = int | float | Fraction


@dataclass
class Polynomial:
    """
    Represents polynomials, to be used as values for parametric models.
    Polynomials are represented as a dictionary with n-dimensional tuples as keys.

    Args:
        terms: terms of the polynomial (dictionary that relates exponents to coefficients)
        variables: variables of the polynomial as a list of strings
    """

    terms: dict[tuple, float]
    variables: list[str]

    def __init__(self, variables: list[str]):
        self.terms = dict()
        self.variables = variables

    def add_term(self, exponents: tuple[int, ...], coefficient: float):
        # TODO exponents may also be a single integer
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
        """returns the number of different variables present"""
        return len(self.variables)

    def get_variables(self) -> set[str]:
        """returns the set of parameters"""
        return set(self.variables)

    def get_degree(self) -> int | None:
        """returns the degree of the polynomial"""
        if self.terms is not {}:
            largest = 0
            for term in self.terms.keys():
                current = sum(list(term))
                if current > largest:
                    largest = current
            return largest
        raise RuntimeError("A polynomial without terms does not have a degree.")

    def evaluate(self, values: dict[str, Number]) -> float:
        """evaluates the polynomial with the given values for the variables"""
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
        return str(self.terms) < str(other.terms)

    def __eq__(self, other) -> bool:
        if isinstance(other, Polynomial):
            return self.terms == other.terms
        return False

    def __iter__(self):
        return iter(self.terms.items())


class RationalFunction:
    """
    Represents rational functions, to be used as values for parametric models
    Rational functions are represented as a pair of polynomials.

     Args:
        numerator: Polynomial in the numerator
        denominator: Polynomial in the denominator
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

    def get_dimension(self) -> int:
        """returns the number of different variables present"""
        return max(self.numerator.get_dimension(), self.denominator.get_dimension())

    def get_variables(self) -> set[str]:
        "returns the total set of variables of this rational function"
        return set(self.numerator.variables).union(set(self.denominator.variables))

    def evaluate(self, values: dict[str, Number]) -> float:
        """evaluates the rational function with the given values"""
        return self.numerator.evaluate(values) / self.denominator.evaluate(values)

    def __str__(self) -> str:
        s = "(" + str(self.numerator) + ")/(" + str(self.denominator) + ")"
        return s

    def __lt__(self, other) -> bool:
        if isinstance(other, Polynomial):
            return self.numerator < other or self.denominator < other
        else:
            return (
                self.numerator < other.numerator or self.denominator < other.denominator
            )


Parametric = Polynomial | RationalFunction
