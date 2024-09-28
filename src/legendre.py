import numpy


def binomial_coefficient(n: int, k: int) -> int:
    return numpy.arange(1, n + 1).prod() / (numpy.arange(1, k + 1).prod() * numpy.arange(1, n - k + 1).prod())


class Legendre:
    def __init__(self, n: int) -> None:
        self._n = n
        self._participants = self.calculate_participants()

    @property
    def order(self) -> int:
        return self._n

    def __call__(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        terms = []
        for participant in self._participants:
            terms.append(participant[0] * numpy.power(x, participant[1]))

        return numpy.array(terms).sum(axis=0)

    def calculate_participants(self) -> list[tuple[float, float]]:
        parts = []
        for k in range(self._n // 2 + 1):
            parts.append((self.coefficient_function(k), self.power_function(k)))
        return parts

    def power_function(self, k: int) -> int:
        return self._n - 2 * k

    def coefficient_function(self, k: int) -> float:
        return 1 / numpy.power(2, self._n) * numpy.power(-1, k) * \
        binomial_coefficient(self._n, k) * binomial_coefficient(2 * self._n - 2 * k, self._n)


if __name__ == '__main__':
    pass
