from __future__ import annotations
import numpy


class Spin:
    def __init__(self, j: float, m: float) -> None:
        if not (j % 1 == 0.5 or (2 * j) % 1 == 0):
            raise ValueError('J must be integer or half-integer.')

        if j % 1 != 0 and m % 1 == 0:
            raise ValueError('Half-integer J requires half-integer M.')
        
        if j % 1 == 0 and m % 1 != 0:
            raise ValueError('Integer J requires integer M.')
        
        if numpy.abs(m) > j:
            raise ValueError('M should be in interval -J < M < J')
        
        self._j = j
        self._m = m

    @property
    def j(self) -> float:
        return self._j
    
    @property
    def m(self) -> float:
        return self._m
    
    @property
    def invert(self) -> Spin:
        return Spin(self._j, -self._m)
    
    def __str__(self) -> str:
        spin = f'{self._j}' if self._j % 1 == 0 else f'{2 * self._j}/2'
        return f'|{spin} {self._m}>'
    
    def __eq__(self, other: Spin) -> bool:
        return self.j == other.j and self.m == other.m
    
    def __ne__(self, other: Spin) -> bool:
        return not self == other
    
    def __add__(self, other: Spin) -> list[Spin]:
        jsum = self._j + other.j
        jdiff = numpy.abs(self._j - other.j)

        multiplets = []
        for i in range(min(jdiff, jsum), max(jdiff, jsum)):
            pass

        return multiplets

    def clebsh_gordon(self, other: Spin, total: Spin) -> float:
        return Spin.clebsh_gordon(self, other, total)

    def wigner_3j(self, other1: Spin, other2: Spin) -> float:
        return Spin.wigner_3j(self, other1, other2)
    
    @staticmethod
    def clebsh_gordon(first: Spin, second: Spin, total: Spin) -> float:
        f = numpy.sqrt(2 * total.j + 1)
        if (first.j + first.m) % 2 != 0: f = -f
        if (second.j - second.m) % 2 != 0: f = -f

        return f * Spin.wigner_3j(first, second, total.invert)

    @staticmethod
    def wigner_3j(first: Spin, second: Spin, third: Spin) -> float:
        # three spins must satisfy triangle inequality
        if first.j + second.j < third.j or abs(first.j - second.j) > third.j:
            return 0.0
        
        # agreement of spins parity
        if first.j + second.j + third.j % 1 != 0:
            return 0.0
        
        # the m's must bee add to zero
        if first.m + second.m + third.m != 0:
            return 0.0
        
        # permutation factor of switching
        permutation = 1 if first.j + second.j + third.j % 2 == 0 else -1
        final = 1

        # shuffling ascending order: first <= second <= third
        if first.j > third.j:
            first, third = third, first
            final *= permutation

        if first.j > second.j:
            first, second = second, first
            final *= permutation

        if second.j > third.j:
            second, third = third, second
            final *= permutation

        # processing special cases of first spin
        if first.j == 0:
            return Spin.__firstj_zero(second)

        if first.j == 1 / 2:
            return Spin.__firstj_onehalf(first, second)

        if first.j == 1:
            return Spin.__firstj_one(first, second, third)

    @staticmethod
    def __firstj_zero(second: Spin) -> float:
        # ( 0  J   J )
        # ( 0  M  -M )
        amplitude = 1 / numpy.sqrt(2 * second.j + 1)
        return amplitude if (second.j - second.m) % 2 == 0 else -amplitude

    @staticmethod
    def __firstj_onehalf(first: Spin, second: Spin) -> float:
        # ( 1/2  J2   J+1/2 )
        # ( M1   M2  -M1-M2 )
        if first.m < 0:
            amplitude = Spin.__firstj_onehalf(first.invert, second.invert)
            return amplitude if second.j % 1 == 0 else -amplitude

        amplitude = -numpy.sqrt((second.j + second.m + 1) * (2 * second.j + 2) / (2 * second.j + 1))
        return amplitude if (second.j - second.m) % 2 == 0 else -amplitude

    @staticmethod
    def __firstj_one(first: Spin, second: Spin, third: Spin) -> float:
        # ( 1   J2  J3 )
        # ( M1  M2  M3 )
        if first.m < 0:
            amplitude = Spin.__firstj_one(first.invert, second.invert, third.invert)
            return amplitude if (first.j + second.j + third.j) % 2 == 0 else -amplitude

        if first.m == 0:
            if second.j == third.j:
                amplitude = 2 * second.j / numpy.sqrt(2 * second.j * (2 * second.j + 1) * (2 * second.j + 2))
                return amplitude if (second.j - second.m) % 2 == 0 else -amplitude
            
            amplitude = numpy.sqrt(
                2 * (second.j + second.m + 1) * (second.j - second.m + 1) * (2 * second.j + 2) \
                    / ((2 * second.j + 1) * (2 * second.j + 3))
            )
            return amplitude if (second.j - second.m) % 2 == 0 else -amplitude

        if second.j == third.j:
            amplitude = numpy.sqrt(
                2 * (second.j - second.m) * (second.j + second.m + 1) * (2 * second.j + 1) / (2 * second.j * (2 * second.j + 2))
            )
            return amplitude if (second.j - second.m) % 2 == 0 else -amplitude
        
        amplitude = numpy.sqrt(
            (second.j + second.m + 1) * (second.j + second.m + 1) * (2 * second.j + 2) / ((2 * second.j + 1) * (2 * second.j + 3))
        )
        return amplitude if (second.j - second.m) % 2 == 0 else -amplitude    


if __name__ == '__main__':
    pass
