import numpy


class Numerov:
    def __init__(self, Veff = lambda r: 1 / r, rmax: float = 25.0, dr: float = 0.01) -> None:
        self.Veff = Veff

        self.rmax = rmax
        self.dr = dr

    def solve(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        grid = numpy.linspace(0, self.rmax, int(self.rmax / self.dr))
        ys = [0.0, 0.1]

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
    import matplotlib.pyplot as plt

    h_bar = 6.582119e-22 # MeV * s
    c = 3e23 # fm / s
    fine_structure = 1 / 137

    m1 = 3 * 938.37 + 3 * 939.57 # MeV
    m2 = 2 * 938.37 + 2 * 939.57 # MeV
    mu = m1 * m2 / (m1 + m2) # MeV
    E_cm = 15 # MeV

    Rc = 1.28 # fm
    charges = 6
    e_power_2 = fine_structure * h_bar * c

    l = 4

    coul_pot = lambda r: -charges * e_power_2 / r if r > Rc else -charges * e_power_2 / Rc * (3 / 2 - r ** 2 / (2 * Rc ** 2))
    real_pot = lambda r: -100 / (1 + numpy.exp((r - numpy.cbrt(6) * 1.28) / 0.7)) # MeV
    imag_pot = lambda r:  -20 / (1 + numpy.exp((r - numpy.cbrt(6) * 1.57) / 0.7)) # MeV

    pot = lambda r: 2 * mu / (h_bar ** 2 * c ** 2) * (l * (l + 1) / r ** 2 + E_cm - complex(real_pot(r), imag_pot(r)) - coul_pot(r))

    num = Numerov(pot, rmax=25, dr=0.01)

    rs, psis = num.solve()

    plt.plot(rs, [psi.real for psi in psis], color='blue')
    plt.plot(rs, [psi.imag for psi in psis], color='red')

    # plt.plot(rs, [psi * psi.conjugate() for psi in psis])

    plt.grid()
    plt.show()
