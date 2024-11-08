import numpy
from mpmath import coulombf, coulombg


def factorial(n: int) -> int:
    return numpy.arange(1, n + 1).prod()


def arg_gamma(z: complex) -> float:
    return numpy.angle(gamma_function(z))


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
    

class Regular:
    def __init__(self, l: int) -> None:
        self.__l = l

    @property
    def l(self) -> int:
        return self.__l

    def __call__(self, etha: float, ro: float) -> float:
        return coulombf(self.__l, etha, ro)
    

class Irregular:
    def __init__(self, l: int) -> None:
        self.__l = l

    @property
    def l(self) -> int:
        return self.__l

    def __call__(self, etha: float, ro: float) -> float:
        return coulombg(self.__l, etha, ro)


class CoulombWaveFunction:
    def __init__(self, l: int, w: bool) -> None:
        self.__l = l
        self.__w = w

    @property
    def l(self) -> int:
        return self.__l
    
    @property
    def w(self) -> bool:
        return self.__w

    def __call__(self, etha: float, ro: float) -> numpy.ndarray:
        return Irregular(self.__l)(etha, ro) + complex(0, 1 if self.__w else -1) * Regular(self.__l)(etha, ro)

    def derivative(self, etha: float, ro: float) -> numpy.ndarray:
        rl = 1 + numpy.sqrt(etha ** 2 / (self.__l + 1) ** 2)
        sl = (self.__l + 1) / ro + etha / (self.__l + 1)
        return rl * CoulombWaveFunction(self.__l + 1, self.__w)(etha, ro) - sl * CoulombWaveFunction(self.__l, self.__w)(etha, ro)


if __name__ == '__main__':
    pass
