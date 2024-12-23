import numpy


class Numerov:
    def __init__(self, Veff = lambda r: 1 / r, rmin: float = 0.0, rmax: float = 25.0, dr: float = 0.01) -> None:
        self.Veff = Veff

        self.rmin = rmin
        self.rmax = rmax
        self.dr = dr

    def solve(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        grid = numpy.linspace(self.rmin, self.rmax, int(self.rmax / self.dr))
        ys = [0.0, self.dr]

        for i in range(2, len(grid)):
            ys.append(self.next_point(ys[i - 1], ys[i - 2], grid[i]))

        return (grid, ys)

    def next_point(self, prev_y: float, prev_prev_y: float, r: float) -> float:
        prev_prev_g = self.Veff(r - self.dr)
        prev_g = self.Veff(r)
        next_g = self.Veff(r + self.dr)

        first_term = 2 * prev_y * (1 - (5 * self.dr ** 2) / 12 * prev_g)
        second_term = prev_prev_y * (1 + self.dr ** 2 / 12 * prev_prev_g)
        denumerator = 1 + self.dr ** 2 / 12 * next_g

        return (first_term - second_term) / denumerator


if __name__ == '__main__':
    pass
