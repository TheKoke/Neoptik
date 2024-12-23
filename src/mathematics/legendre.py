import numpy


class Legendre:
    def __init__(self, n: int) -> None:
        self._n = n

    @property
    def order(self) -> int:
        return self._n

    def __call__(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        pn = [1, x]
        for n in range(2, self.order + 1):
            prev = (2 * n + 1) / (n + 1) * x * pn[n - 1]
            prev_prev = n / (n + 1) * pn[n - 2]

            pn.append(prev - prev_prev)

        return pn[self.order]


if __name__ == '__main__':
    pass
