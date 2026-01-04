import numpy


def factorial(n: int) -> int:
    return numpy.arange(1, n + 1).prod()


def coulomb_phase_shift(l: int, eta: float) -> float:
    return numpy.angle(numpy.exp(1j * numpy.log(numpy.abs(gamma_function(1j * eta))))) + sum([numpy.arctan(eta / n) for n in range(1, l + 1)])


def gamma_function(z: complex) -> complex:
    lanczos_parameter = 7
    lanczos_coeffs = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ]

    if z * z.conjugate() == 0:
        return 1

    if z.real < 0.5:
        return numpy.pi / (numpy.sin(numpy.pi * z) * gamma_function(1 - z))  # Reflection formula
    else:
        z -= 1

        x = lanczos_coeffs[0]
        for i in range(1, len(lanczos_coeffs)):
            x += lanczos_coeffs[i] / (z + i)

        sqrt = numpy.sqrt(2 * numpy.pi)
        power = numpy.power(z + lanczos_parameter + 1/2, z + 1/2)
        exp = numpy.exp(-(z + lanczos_parameter + 1/2))

    return sqrt * power * exp * x
    

class CoulombWaveFunctions:
    def __init__(self, l: int) -> None:
        self.__l = l

        self.__etha = 0.0
        self.__ro = 0.0
        self.__fl = -999.0 * numpy.ones(1) 
        self.__gl = -999.0 * numpy.ones(1) 
        self.__hlp = -999.0 * numpy.ones(1) 
        self.__hlm = -999.0 * numpy.ones(1) 
        self.__dfl = -999.0 * numpy.ones(1) 
        self.__dgl = -999.0 * numpy.ones(1) 
        self.__dhlp = -999.0 * numpy.ones(1) 
        self.__dhlm = -999.0 * numpy.ones(1)

    @property
    def l(self) -> int:
        return self.__l
    
    def regular(self, etha: float, ro: numpy.ndarray) -> numpy.ndarray: 
        if self.__etha != etha or (len(self.__fl) == 1 and self.__fl[0] == -999.0): 
            self.__compute(etha, ro) 
        return self.__fl 
        
    def irregular(self, etha: float, ro: numpy.ndarray) -> numpy.ndarray: 
        if self.__etha != etha or (len(self.__gl) == 1 and self.__gl[0] == -999.0): 
            self.__compute(etha, ro) 
        return self.__gl 
    
    def hplus(self, etha: float, ro: numpy.ndarray) -> numpy.ndarray: 
        if self.__etha != etha or (len(self.__hlp) == 1 and self.__hlp[0] == -999.0): 
            self.__compute(etha, ro) 
        return self.__hlp 
    
    def hminus(self, etha: float, ro: numpy.ndarray) -> numpy.ndarray: 
        if self.__etha != etha or (len(self.__hlm) == 1 and self.__hlm[0] == -999.0): 
            self.__compute(etha, ro) 
        return self.__hlm 
    
    def dregular(self, etha: float, ro: numpy.ndarray) -> numpy.ndarray: 
        if self.__etha != etha or (len(self.__dfl) == 1 and self.__dfl[0] == -999.0): 
            self.__compute(etha, ro) 
        return self.__dfl 
    
    def dirregular(self, etha: float, ro: numpy.ndarray) -> numpy.ndarray: 
        if self.__etha != etha or (len(self.__dgl) == 1 and self.__dgl[0] == -999.0): 
            self.__compute(etha, ro) 
        return self.__dgl 
    
    def dhplus(self, etha: float, ro: numpy.ndarray) -> numpy.ndarray: 
        if self.__etha != etha or (len(self.__dhlp) == 1 and self.__dhlp[0] == -999.0): 
            self.__compute(etha, ro) 
        return self.__dhlp 
    
    def dhminus(self, etha: float, ro: numpy.ndarray) -> numpy.ndarray: 
        if self.__etha != etha or (len(self.__dhlm) == 1 and self.__dhlm[0] == -999.0): 
            self.__compute(etha, ro) 
        return self.__dhlm

    def _potential(self, eta: float, ro: float) -> float:
        return 1.0 - 2.0 * eta / ro - self.l * (self.l + 1) / ro ** 2

    def _integrate_outward(self, eta: float | complex, ro_grid: numpy.ndarray) -> numpy.ndarray:
        h = ro_grid[1] - ro_grid[0]
        u = numpy.zeros_like(ro_grid)

        # Small-r behavior: F_l ~ ro^{l+1}
        u[0] = ro_grid[0] ** (self.l + 1)
        u[1] = ro_grid[1] ** (self.l + 1)

        for i in range(1, len(ro_grid) - 1):
            k_im1 = self._potential(eta, ro_grid[i - 1])
            k_i   = self._potential(eta, ro_grid[i])
            k_ip1 = self._potential(eta, ro_grid[i + 1])

            u[i + 1] = (
                (2 * (1 - 5 * h**2 * k_i / 12) * u[i]
                 - (1 + h**2 * k_im1 / 12) * u[i - 1])
                / (1 + h**2 * k_ip1 / 12)
            )

        return u

    def _integrate_inward(self, eta: float | complex, ro_grid: numpy.ndarray) -> numpy.ndarray:
        h = ro_grid[1] - ro_grid[0]
        u = numpy.zeros_like(ro_grid)

        # Asymptotic form for large ro
        phase = ro_grid[-1] - eta * numpy.log(2 * ro_grid[-1]) - self.l * numpy.pi / 2
        u[-1] = numpy.cos(phase)
        u[-2] = numpy.cos(ro_grid[-2] - eta * numpy.log(2 * ro_grid[-2]) - self.l * numpy.pi / 2)

        for i in range(len(ro_grid) - 2, 0, -1):
            k_ip1 = self._potential(eta, ro_grid[i + 1])
            k_i   = self._potential(eta, ro_grid[i])
            k_im1 = self._potential(eta, ro_grid[i - 1])

            u[i - 1] = (
                (2 * (1 - 5 * h**2 * k_i / 12) * u[i]
                 - (1 + h**2 * k_ip1 / 12) * u[i + 1])
                / (1 + h**2 * k_im1 / 12)
            )

        return u

    def __compute(self, eta: float, ro: numpy.ndarray) -> None:
        ro = numpy.asarray(ro)
        assert numpy.all(ro > 0), "ro must be positive"

        F = self._integrate_outward(eta, ro)
        G = self._integrate_inward(eta, ro)

        i = len(ro) // 2
        dF = (F[i + 1] - F[i - 1]) / (ro[i + 1] - ro[i - 1])
        dG = (G[i + 1] - G[i - 1]) / (ro[i + 1] - ro[i - 1])

        W = F[i] * dG - dF * G[i]
        G /= W

        dF = numpy.gradient(F, ro)
        dG = numpy.gradient(G, ro)

        H_plus  = F + 1j * G
        H_minus = F - 1j * G

        dH_plus  = dG + 1j * dF
        dH_minus = dG - 1j * dF

        self.__fl = F; self.__gl = G
        self.__dfl = dF; self.__dgl = dG
        self.__hlp = H_plus; self.__hlm = H_minus
        self.__dhlp = dH_plus; self.__dhlm = dH_minus


if __name__ == '__main__':
    pass
