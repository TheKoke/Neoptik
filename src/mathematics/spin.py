from __future__ import annotations
import numpy


def logfactorial(n: float) -> float:
    # Ramanujan's approximation formula for log(n!)
    return n * numpy.log(n) - n + numpy.log(8 * n ** 3 + 4 * n ** 2 + n) / 6 + numpy.log(numpy.pi) / 2


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
        spin = f'{int(self._j)}' if self._j % 1 == 0 else f'{int(2 * self._j)}/2'
        momentum = f'{int(self._m)}' if self._m % 1 == 0 else f'{int(2 * self._m)}/2'
        return f'|{spin} {momentum}>'
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other: Spin) -> bool:
        return self.j == other.j and self.m == other.m
    
    def __ne__(self, other: Spin) -> bool:
        return not self == other
    
    def __add__(self, other: Spin) -> list[Spin]:
        jsum = self._j + other.j
        jdiff = numpy.abs(self._j - other.j)
        possible_js = [min(jdiff, jsum) + i for i in range(max(jdiff, jsum) - min(jdiff, jsum))]

        multiplets = []
        for i in possible_js:
            try:
                multiplets.append(Spin(i, self.m + other.m))
            except ValueError as e:
                print(e)

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
        if (first.j + second.j + third.j) % 1 != 0:
            return 0.0
        
        # the m's must sum to zero
        if (first.m + second.m + third.m) != 0:
            return 0.0
        
        # permutation factor of switching
        permutation = 1 if (first.j + second.j + third.j) % 2 == 0 else -1
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
        
        if first.j == 3 / 2:
            return Spin.__firstj_threehalf(first, second, third)

        if first.m == 0 and second.m == 0 and third.m == 0:
            return Spin.__allm_zero(first, second, third)

        return Spin.shulten_gordon(first, second, third)

    @staticmethod
    def __firstj_zero(second: Spin) -> float:
        # ( 0  J   J )
        # ( 0  M  -M )
        amplitude = 1 / numpy.sqrt(2 * second.j + 1)
        return amplitude if (second.j - second.m) % 2 == 0 else -amplitude

    @staticmethod
    def __firstj_onehalf(first: Spin, second: Spin) -> float:
        # ( 1/2  J2  J2+1/2 )
        # ( M1   M2  -M1-M2 )
        if first.m < 0:
            amplitude = Spin.__firstj_onehalf(first.invert, second.invert)
            return amplitude if second.j % 1 == 0 else -amplitude

        amplitude = -numpy.sqrt((second.j + second.m + 1) / (2 * second.j + 1) / (2 * second.j + 2))
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
                amplitude = 2 * second.m / numpy.sqrt(2 * second.j * (2 * second.j + 1) * (2 * second.j + 2))
                return amplitude if (second.j - second.m) % 2 == 0 else -amplitude
            
            amplitude = numpy.sqrt(
                2 * (second.j + second.m + 1) * (second.j - second.m + 1) / (2 * second.j + 1) / (2 * second.j + 2) / (2 * second.j + 3)
            )
            return amplitude if (second.j - second.m) % 2 == 0 else -amplitude

        if second.j == third.j:
            amplitude = numpy.sqrt(
                2 * (second.j - second.m) * (second.j + second.m + 1) / (2 * second.j) / (2 * second.j + 1) / (2 * second.j + 2)
            )
            return amplitude if (second.j - second.m) % 2 == 0 else -amplitude
        
        amplitude = numpy.sqrt(
            (second.j + second.m + 1) * (second.j + second.m + 2) / (2 * second.j + 1) / (2 * second.j + 2) / (2 * second.j + 3)
        )
        return amplitude if (second.j - second.m) % 2 == 0 else -amplitude
    
    @staticmethod
    def __firstj_threehalf(first: Spin, second: Spin, third: Spin) -> float:
        # ( 3/2  J2  J3 )
        # ( M1   M2  M3 )
        if first.m < 0:
            return Spin.__firstj_threehalf(first.invert, second.invert, third.invert)

        if first.m == 1 / 2:
            if third.j == second.j + 1 / 2:
                amplitude = (second.j - 3 * second.m) * numpy.sqrt(
                    (second.j + second.m + 1) / (2 * second.j) / (2 * second.j + 1) / (2 * second.j + 2) / (2 * second.j + 3)
                )
                return amplitude if (second.j - second.m) % 2 == 0 else -amplitude

            amplitude = numpy.sqrt(
                3 * (second.j + second.m + 1) * (second.j + second.m + 2) * (second.j - second.m + 1) / (2 * second.j + 1) / (2 * second.j + 2) / (2 * second.j + 3) / (2 * second.j + 4)
            )
            return amplitude if (second.j - second.m) % 2 == 0 else -amplitude
        
        if third.j == second.j + 1 / 2:
            amplitude = -numpy.sqrt(
                3 * (second.j + second.m + 1) * (second.j + second.m + 2) * (second.j - second.m) / (2 * second.j) / (2 * second.j + 1) / (2 * second.j + 2) / (2 * second.j + 3)
            )
            return amplitude if (second.j - second.m) % 2 == 0 else -amplitude

        amplitude = -numpy.sqrt(
            (second.j + second.m + 1) * (second.j + second.m + 2) * (second.j + second.m + 3) / (2 * second.j + 1) / (2 * second.j + 2) / (2 * second.j + 3) / (2 * second.j + 4)
        )
        return amplitude if (second.j - second.m) % 2 == 0 else -amplitude
    
    @staticmethod
    def __allm_zero(first: Spin, second: Spin, third: Spin) -> float:
        # ( J1  J2  J3 )
        # ( 0   0   0  )
        summary = first.j + second.j + third.j
        if summary % 2 != 0:
            return 0.0
        
        f = -logfactorial(summary + 1) \
            + logfactorial(summary - 2 * first.j) \
            + logfactorial(summary - 2 * second.j) \
            + logfactorial(summary - 2 * third.j)
        f = numpy.exp(f / 2)

        g = logfactorial(summary / 2) \
            - logfactorial(summary / 2 - first.j) \
            - logfactorial(summary / 2 - second.j) \
            - logfactorial(summary / 2 - third.j)
        g = numpy.exp(g)

        return f * g if summary % 4 == 0 else - (f * g)

    @staticmethod
    def shulten_gordon(first: Spin, second: Spin, third: Spin) -> float:
        """
        Recursive algorithm of calculation Wigner's 3j-symbol.\n
        Ref. 
        ----
        Schulten, K., & Gordon, R. G. (1975). 
        Exact recursive evaluation of 3j- and 6j-coefficients for quantum-mechanical coupling of angular momenta.
        Journal of Mathematical Physics, 16(10), 1961-1970.
        """
        # the borders of recursion
        jmin = max(abs(2 * first.j - 2 * second.j), abs(2 * third.m))
        jmax = 2 * first.j + 2 * second.j

        # middle point
        jmid = (jmax + jmin) // 2
        jmid = jmid if jmid % 2 == jmax % 2 else jmid + 1

        # initializing the coefficients of recurrence relations (A & B)
        prev_a = 0.0
        curr_a = 0.0
        curr_b = 0.0

        # the states of C coefficient
        curr_c = (-1) ** ((first.j - second.j - third.m) % 2) * -0.5 / jmax
        next_c = 0.0
        prev_c = 0.0
        target_c = 0.0

        j = jmax
        norm_right = 0.0

        # Recursion from right: from jmax to jmid
        while True:
            norm_right += (j + 1) * curr_c * curr_c

            # Memorizing C of target j
            if j == 2 * third.j:
                target_c = curr_c

            # Accuracy check and adaptive shift of jmid
            if j <= jmid:
                if abs(curr_c) <= 2 ** (-40) * abs(next_c):
                    jmid -= 2
                else:
                    break

            curr_a = Spin.__shulten_gordon_a(2 * first.j, 2 * second.j, j, 2 * third.m)
            curr_b = Spin.__shulten_gordon_b(2 * first.j, 2 * second.j, j, 2 * first.m, 2 * second.m, 2 * third.m)

            # Recurrent relation for C
            prev_c = (-curr_b * curr_c - j / 2 * prev_a * next_c) / ((j + 2) / 2) / curr_a

            next_c = curr_c
            curr_c = prev_c
            prev_a = curr_a
            j -= 2

        # Recursion from left: from jmin to jmid
        norm_left = 0.0
        saved_c_at_mid = curr_c
        prev_c = 0.0
        curr_a = 0.0
        curr_c = jmax / 2
        j = jmin

        # Exception j = 0
        if j == 0:
            norm_left += curr_c * curr_c
            prev_c = curr_c
            curr_c = 2 * first.m * prev_c / numpy.sqrt(2 * first.j * (2 * first.j + 2))
            curr_a = Spin.__shulten_gordon_a(2 * first.j, 2 * second.j, 2, 0)
            j = 2

        while True:
            if j >= jmid:
                break

            if j == 2 * third.j:
                target_c = curr_c

            norm_left += (j + 1) * curr_c * curr_c

            if j >= jmax:
                break

            next_a = Spin.__shulten_gordon_a(2 * first.j, 2 * second.j, j + 2, 2 * third.m)
            curr_b = Spin.__shulten_gordon_b( 2 * first.j, 2 * second.j, j, 2 * first.m, 2 * second.m, 2 * third.m)

            next_c = (-curr_b * curr_c - (j + 2) / 2 * curr_a * prev_c) / (j / 2) / next_a

            prev_c = curr_c
            curr_c = next_c
            curr_a = next_a
            j += 2

        # Normalization
        ratio = curr_c / saved_c_at_mid
        norm_left /= ratio ** 2

        if 2 * third.j < jmid:
            target_c /= ratio

        normalization = norm_left + norm_right
        return target_c / numpy.sqrt(normalization)

    @staticmethod
    def __shulten_gordon_a(j1: float, j2: float, j3: float, m3: float) -> float:
        f1 = (j3 - (j1 - j2)) * (j3 + (j1 - j2)) / 4
        f2 = (j1 + j2 - j3 + 2) * (j1 + j2 + j3 + 2) / 4
        f3 = (j3 + m3) * (j3 - m3) / 4
        return numpy.sqrt(f1 * f2 * f3)

    @staticmethod
    def __shulten_gordon_b(j1: float, j2: float, j3: float, m1: float, m2: float, m3: float) -> float:
        f = (m3 * (j1 * (j1 + 2) - j2 * (j2 + 2)) + (m1 - m2) * j3 * (j3 + 2)) / 8
        return -(j3 + 1) * f


if __name__ == '__main__':
    pass
