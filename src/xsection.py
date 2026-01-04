import numpy
import multiprocessing

from nuclear import Nuclei
from mathematics.chi import Chi_Square
from mathematics.numerov import Numerov
from mathematics.legendre import Legendre
from mathematics.couloumb import CoulombWaveFunctions, coulomb_phase_shift
from potentials import Optical, WSVolume, WSSurface, WSParameters, Coulomb, SpinOrbit


class Elastic:
    def __init__(self, potential: Optical, energy: float) -> None:
        self._potential = potential
        self._beam = potential.beam
        self._target = potential.target
        self._energy = energy

    @property
    def beam(self) -> Nuclei:
        '''
        Returns
        -------
        `beam` : `Nuclei`
            Beam nuclei in nuclear reaction.
        '''
        return self._beam
    
    @property
    def target(self) -> Nuclei:
        '''
        Returns
        -------
        `target` : `Nuclei`
            Target nuclei in nuclear reaction.
        '''
        return self._target
    
    @property
    def energy(self) -> float:
        '''
        Returns
        -------
        `energy` : `float`
            Energy of beam in lab. system, MeV.
        '''
        return self._energy
    
    @property
    def center_mass_energy(self) -> float:
        '''
        Returns
        -------
        `Ecm` : `float`
            Energy of system in c.m., MeV.
        '''
        return self._energy * (1 - self._beam.mass() / (self._beam.mass() + self._target.mass())) # MeV
    
    @property
    def reduced_mass(self) -> float:
        '''
        Returns
        -------
        `mu` : `float`
            Reduced mass of interacting nucleus, MeV.
        '''
        return self._beam.mass() * self._target.mass() / (self._beam.mass() + self._target.mass()) # MeV
    
    @property
    def wavenumber(self) -> float:
        '''
        Returns
        -------
        `wavenumber` : `float`
            Wavenumber of system in c.m., fm^(-1).
        '''
        h_bar = 6.582119e-22 # MeV * s
        c = 3e23 # fm / s
        return numpy.sqrt(2 * self.reduced_mass * self.center_mass_energy / (h_bar ** 2 * c ** 2)) # fm^(-1)
    
    @property
    def sommerfield(self) -> float:
        '''
        Returns
        -------
        `etha` : `float`
            Somerfield parameter of interacting nuclei, dimensionless.
        '''
        z1 = self._beam.charge; z2 = self._target.charge
        fine_structure = 1 / 137 # dimensionless
        return z1 * z2 * fine_structure * numpy.sqrt(self.reduced_mass / (2 * self.center_mass_energy)) # dimensionless
    
    @property
    def potential(self) -> Optical:
        '''
        Returns
        -------
        `potetnial` : `Optical`
            Optical potential for calculating cross-sections.
        '''
        return self._potential
    
    def xsections(self, theta0: float, thetan: float, dtheta: float,
                  lmax: int = 20, 
                  rmax: float = 30.0, dr: float = 0.01) -> tuple[numpy.ndarray, numpy.ndarray]:
        '''
        Main method for calculating elastic cross-section for given reaction.

        Params
        ------
        `theta0` : `float`
            Angle, calculating elastic cross-section starts from, deg.

        `thetan` : `float`
            Angle, calculating elastic cross-section ends, deg.

        `dtheta` : `float`
            Angle step-size, deg.

        `lmax` : `int`
            Count of partial waves for sum, encounts from 0. Default value = 20.

        `rmax` : `float`
            Maximal radius for integration, fm. Default value = 30.0.

        `dr` : `float`
            Integration step size, fm. Default value = 0.01.

        Returns
        -------

        `angles, xsections` : `tuple[numpy.ndarray, numpy.ndarray]`
            Elastic cross-section. Angles in degrees, xsections in mb/sr.
        '''
        THREADS = lmax + 1

        angles = numpy.linspace(theta0, thetan, int((thetan - theta0) / dtheta) + 1)
        amplitudes = numpy.zeros_like(angles, dtype=numpy.complex64)

        results = []
        # with multiprocessing.Pool(THREADS) as pool:
        #     results = pool.starmap(self.partial_wave_amplitude, [(angles, i, rmax, dr) for i in range(THREADS)])
        for i in range(THREADS):
            results.append(self.partial_wave_amplitude(angles, i, rmax, dr))

        amplitudes += numpy.array(results).sum(axis=0)
        amplitudes += self.rutherford_amplitude(angles)
        cross = (amplitudes * amplitudes.conj()).real

        return angles, cross
    
    def partial_wave_amplitude(self, angles: numpy.ndarray, l: int, rmax: float, dr: float) -> numpy.ndarray:
        '''
        Method that calculates certain partial wave amplitude - `fl`

        Params
        ------
        `angles` : `numpy.ndarray[float]`
            Angles amplitude calculates for, deg.

        `l` : `int`
            Partial wave number.

        `rmax` : `float`
            Maximal radius for integration, fm.

        `dr` : `float`
            Integration step size, fm.

        Returns
        -------
        `fl` : `numpy.ndarray[complex]`
            Certain partial wave amplitude, (mb/sr)^(1/2).
        '''
        rmin = 2 * l * dr if l > 0 else dr
        grid = numpy.linspace(rmin, rmax, int((rmax - rmin) / dr) + 1)

        solutions = self.radial_solutions(l, grid)
        smatrix = self.smatrix(l, solutions)
        print(smatrix)

        legendre = Legendre(l)
        return 1 / (complex(0, 2 * self.wavenumber)) * (2 * l + 1) * legendre(numpy.cos(numpy.radians(angles))) * (smatrix - 1)
    
    def radial_solutions(self, l: int, r: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        '''
        Method for numerical solving Schrodinger equations with given potential `self.potential`

        Params
        ------
        `l` : `int`
            Partial wave number.\n

        `r` : `numpy.ndarray`
            Integration grid.

        Returns
        --------
        `radials` : `tuple[numpy.ndarray[complex], numpy.ndarray[complex]]`
            Solution of Schrodinger eq. 
        '''
        c = 3e23 # fm / s
        h_bar = 6.582119e-22 # MeV * s
        mu = self.reduced_mass
        ecm = self.center_mass_energy
        potential = l * (l + 1) / (r ** 2) + 2 * mu / (h_bar ** 2 * c ** 2) * (self.potential(r) - ecm)

        return Numerov(potential, r).solve()

    def outward_radiuses(self) -> tuple[float, float]:
        '''
        Method that \'calculates\' outward radiuses `a`.

        Returns
        -------
        `radius` : `float`
            Matching outward radius in fermi.
        '''
        return (11.6, 11.7)
    
    def smatrix(self, l: int, solutions: tuple[numpy.ndarray, numpy.ndarray]) -> complex:
        '''
        Params
        ------
        `a` : `float`
            Matching radius, fm.

        `l` : `int`
            Partial wave number.

        `solutions` : `tuple[numpy.ndarray[complex], numpy.ndarray[complex]]`
            Solution of Schrodinger equation.

        Returns
        -------
        `S` : `complex`
            Scaterring matrix of certain partial wave, dimensionless.
        '''
        k = self.wavenumber
        etha = self.sommerfield
        a1, a2 = self.outward_radiuses()
    
        index1 = numpy.abs(solutions[0] - a1).argmin()
        index2 = numpy.abs(solutions[0] - a2).argmin()

        xl1 = solutions[1][index1]
        xl2 = solutions[1][index2]
        relation = xl1 / xl2

        cf = CoulombWaveFunctions(l)
        hplus = cf.hplus(etha, k * solutions[0])
        hminus = cf.hminus(etha, k * solutions[0])

        numerator = relation * hminus[index2] - hminus[index1]
        denumerator = relation * hplus[index2] - hplus[index1]

        smatrix = numerator / denumerator

        return complex(smatrix)
    
    def rutherford_amplitude(self, thetas: numpy.ndarray) -> numpy.ndarray:
        '''
        Params
        ------
        `thetas` : `numpy.ndarray`
            Angles Rutherford scattering calculates to, deg.

        Returns
        -------
        `fc` : `numpy.ndarray[float]`
            Coulomb scattering amplitude, (mb/sr)^(1/2)
        '''
        const = - self.sommerfield / (2 * self.wavenumber * numpy.sin(numpy.radians(thetas / 2)) ** 2)
        exp = -1j * self.sommerfield * numpy.log(numpy.sin(numpy.radians(thetas / 2)) ** 2) + 2j * coulomb_phase_shift(0, 1 + 1j * self.sommerfield)

        return const * numpy.exp(exp)

    def chi_square(self, theory: numpy.ndarray, experimenthal: numpy.ndarray, uncertainty: numpy.ndarray) -> float:
        '''
        Params
        ------
        `theory` : `numpy.ndarray[float]`
            Theoretical calculated cross-sections, mb/sr.

        `experimenthal` : `numpy.ndarray[float]`
            Experimenthal measured cross-sections, mb/sr.
        
        `uncertainty` : `numpy.ndarray[float]`
            Uncertatinty of experimenthal cross-sections, dimensionless.

        Returns
        -------
        `chi-square` : `float`
            Chi-square for calculation.
        '''
        return Chi_Square.averaged_chi_square(theory, experimenthal, uncertainty)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    beam = Nuclei(1, 2)
    target = Nuclei(6, 13)
    E_lab = 14.5

    real = WSVolume(beam, target, WSParameters(99.03, 1.20, 0.755))
    imag = WSVolume(beam, target, WSParameters(20.96, 1.31, 0.645), is_imag=True)
    coul = Coulomb(beam, target, 1.30)

    opt = Optical(beam, target, real, imag, coul)

    fig, axes = plt.subplots(1, 2)

    rs = numpy.linspace(0, 15, 100)
    axes[0].plot(rs, [opt.real_part.function(r) for r in rs], color='blue')
    axes[0].plot(rs, [opt.imaginary_part.function(r).imag for r in rs], color='red')
    axes[0].grid()

    elastic = Elastic(opt, E_lab)
    angles, cross = elastic.xsections(10, 180, 0.5, lmax=20)

    # exp_ang, exp_xs = [], []
    # with open('src/exp.txt', 'r') as file:
    #     buffer = file.read().split('\n')
    #     for line in buffer:
    #         exp_ang.append(float(line.split()[0]))
    #         exp_xs.append(float(line.split()[1]))
    
    # thr_ang, thr_xs = [], []
    # with open('src/plot.txt', 'r') as file:
    #     buffer = file.read().split('\n')
    #     for line in buffer:
    #         thr_ang.append(float(line.split()[0]))
    #         thr_xs.append(float(line.split()[1]))

    axes[1].plot(angles, cross, color='blue')
    # axes[1].scatter(exp_ang, exp_xs, color='black')
    # axes[1].plot(thr_ang, thr_xs, color='red')
    axes[1].set_yscale('log')
    axes[1].grid()
    plt.show()