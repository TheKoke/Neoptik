import numpy
from math import factorial


class Legendre:
    def __init__(self, n: int) -> None:
        self._n = n
        self._coefficients = self._calculate_coefficients()
    
    @property
    def order(self) -> int:
        return self._n
    
    def _calculate_coefficients(self) -> list[float]:
        n = self._n
        coefficients = [0.0] * (n + 1)
        
        prefactor = 1.0 / (2 ** n)
        
        for k in range(0, n // 2 + 1):
            term_coeff = (-1) ** k * factorial(n) / (factorial(k) * factorial(n - k))
            term_coeff *= factorial(2 * n - 2 * k) / (factorial(n) * factorial(n - 2 * k))
            
            power = n - 2 * k
            coefficients[power] = prefactor * term_coeff
        
        return coefficients
    
    def __call__(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = numpy.zeros_like(x)
        
        for power, coeff in enumerate(self._coefficients):
            if abs(coeff) > 1e-10:  # Avoid adding negligible terms
                result += coeff * (x ** power)
        
        return result


if __name__ == '__main__':
    pass
