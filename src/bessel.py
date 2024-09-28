import numpy
from integration import Simpson


def factorial(n: int) -> int:
    return numpy.arange(1, n + 1).prod()


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
        if self._n < 0:
            return (-1) ** (-self._n) * BesselJ(-self._n)(x)

        a = 0; b = numpy.pi
        func = lambda theta: numpy.cos(x * numpy.sin(theta) - self._n * theta)

        return 1 / numpy.pi * Simpson.take_integral(func, a, b)
    

class BesselY:
    def __init__(self, order: int) -> None:
        self._n = order

    def __call__(self, x: float | complex) -> float | complex:
        if self._n < 0:
            return (-1) ** (-self._n) * BesselY(-self._n)
        
        a1 = 0; b1 = numpy.pi
        func1 = lambda theta: numpy.sin(x * numpy.sin(theta) - self._n * theta)

        a2 = 0; b2 = 20 # Experiment-known boundary
        func2 = lambda t: (numpy.exp(self._n * t) + (-1) ** self._n * numpy.exp(-self._n * t)) * numpy.exp(-x * numpy.sinh(t))

        return 1 / numpy.pi * (Simpson.take_integral(func1, a1, b1) - Simpson.take_integral(func2, a2, b2))
    

class SphericalBesselJ:
    def __init__(self, order: int) -> None:
        self._n = order

    def __call__(self, x: float | complex) -> float | complex:
        return numpy.sqrt(numpy.pi / (2 * x)) * BesselJ(self._n + 1 / 2)(x)
    

class SphericalBesselY:
    def __init__(self, order: int) -> None:
        self._n = order

    def __call__(self, x: float | complex) -> float | complex:
        return numpy.sqrt(numpy.pi / (2 * x)) * BesselY(self._n + 1 / 2)(x)


if __name__ == '__main__':
    pass
