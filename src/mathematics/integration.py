import numpy

class Simpson:
    '''
    a
    ⌠         (b - a)               n/2-1               n/2
    |f(x)dx = ——————— * [f(a) + 2 *   Σ  f(a+2ih) + 4 *  Σ f(a+(2i-1)h) + f(b)]
    ⌡           3n                   i=1                i=1
    b
    '''

    @staticmethod
    def take_integral(func, a: float, b: float, n: int = 1000) -> float:
        h = (b - a) / n
        nodes = numpy.linspace(a + h, b - h, n - 2)
        
        odd = Simpson.odd_nodes(func, nodes)
        even = Simpson.even_nodes(func, nodes)

        return (b - a) / (3 * n) * (func(a) + 2 * even + 4 * odd + func(b))
    
    @staticmethod
    def odd_nodes(func, nodes: numpy.ndarray) -> float:
        return func(nodes[1:len(nodes):2]).sum()

    @staticmethod
    def even_nodes(func, nodes: numpy.ndarray) -> float:
        return func(nodes[:len(nodes):2]).sum()
    

if __name__ == '__main__':
    pass
