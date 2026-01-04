import numpy


class Numerov:
    def __init__(self, Veff: numpy.ndarray, rgrid: numpy.ndarray) -> None:
        if len(Veff) != len(rgrid):
            raise ValueError("The sizes of potential and integrating grid must be same!")

        self.__Veff = Veff
        self.__grid = rgrid

    @property
    def Veff(self) -> numpy.ndarray:
        return self.__Veff.copy()
    
    @property
    def grid(self) -> numpy.ndarray:
        return self.__grid.copy()

    def solve(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        dr = self.__grid[1] - self.__grid[0]
        ys = [0.0, dr]

        for i in range(2, len(self.__grid) - 1):
            prev_prev_g = self.Veff[i - 1]
            prev_g = self.Veff[i]
            next_g = self.Veff[i + 1]

            first_term = 2 * ys[i - 1] * (1 - prev_g * (5 * dr ** 2) / 12)
            second_term = ys[i - 2] * (1 + prev_prev_g * dr ** 2 / 12)
            denumerator = 1 + next_g * dr ** 2 / 12

            ys.append((first_term - second_term) / denumerator)

        return (self.grid, numpy.array(ys))
    
    def thorlacius(self) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        dr = self.__grid[1] - self.__grid[0]
        ws = [0.0, dr ** 3]
        ys = [0.0, 12 * dr / (self.__Veff[1])]
        dys = []
        
        for i in range(2, len(self.__grid) - 1):
            first_term = 2 * numpy.cosh(numpy.sqrt(self.Veff[i] * dr ** 2)) * ws[i - 1]
            second_term = ws[i - 2]

            ws.append(first_term - second_term)
            ys.append(ws[-1] / (self.__Veff[i] * dr ** 2 / 12))

        for i in range(1, len(ys) - 1):
            first_term = (1 - self.__Veff[i + 1] * dr ** 2 / 6) * ys[i + 1]
            second_term = (1 - self.__Veff[i - 1] * dr ** 2 / 6) * ys[i - 1]
            
            S = self.__Veff[i] * dr ** 2
            denumerator = (dr / 3) * (6 - S) * numpy.sinh(numpy.sqrt(S)) / numpy.sqrt(S)

            dys.append((first_term - second_term) / denumerator)

        return (self.__grid, numpy.array(ys), numpy.array(dys))


if __name__ == '__main__':
    pass
