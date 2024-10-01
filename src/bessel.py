import numpy
from integration import Simpson


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


class BesselJ:
    def __init__(self, order: float) -> None:
        self._n = order

    def __call__(self, x: float | complex) -> float | complex:
        a = 0; b = numpy.pi
        func = lambda theta: numpy.cos(x * numpy.sin(theta) - self._n * theta)

        return 1 / numpy.pi * Simpson.take_integral(func, a, b)
    

class SphericalBesselJ:
    def __init__(self, order: int) -> None:
        self._n = order

    def __call__(self, x: float | complex) -> float | complex:
        return numpy.sqrt(numpy.pi / (2 * x)) * BesselJ(self._n + 1 / 2)(x)
    

class SphericalBesselY:
    def __init__(self, order: int) -> None:
        self._n = order

    def __call__(self, x: float | complex) -> float | complex:
        return (-1) ** (self._n + 1) * numpy.sqrt(numpy.pi / (2 * x)) * BesselJ(-self._n - 1 / 2)(x)
    

class Hankel:
    def __init__(self, order: int, w: bool) -> None:
        self._n = order
        self._w = w

    def __call__(self, x: float | complex) -> complex:
        return SphericalBesselY(self._n)(x) + complex(0, SphericalBesselJ(self._n)(x)) if self._w \
            else SphericalBesselY(self._n)(x) - complex(0, SphericalBesselJ(self._n)(x))


if __name__ == '__main__':
    pass
