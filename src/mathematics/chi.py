import numpy


class Chi_Square:
    
    @staticmethod
    def chi_square(theory: numpy.ndarray, exp: numpy.ndarray, uncertainty: numpy.ndarray) -> numpy.ndarray:
        return numpy.power((theory - exp) / uncertainty, 2)

    @staticmethod
    def averaged_chi_square(theory: numpy.ndarray, exp: numpy.ndarray, uncertainty: numpy.ndarray) -> numpy.ndarray:
        return Chi_Square.chi_square(theory, exp, uncertainty) / len(exp)

    @staticmethod
    def normalized_chi_square(theory: numpy.ndarray, exp: numpy.ndarray, uncertainty: numpy.ndarray) -> numpy.ndarray:
        pass


if __name__ == '__main__':
    pass
