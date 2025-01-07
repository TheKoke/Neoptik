import numpy
import multiprocessing

from nuclear import Nuclei
from potentials import Optical
from mathematics.chi import Chi_Square
from mathematics.numerov import Numerov
from mathematics.legendre import Legendre
from mathematics.couloumb import CoulombWaveFunction, arg_gamma


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
                  rmax: float = 30.0, dr: float = 0.1) -> tuple[numpy.ndarray, numpy.ndarray]:
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
        for i in range(lmax + 1):
            results.append(self.partial_wave_amplitude(angles, i, rmax, dr))

        amplitudes += numpy.array(results).sum(axis=0)
        cross = (amplitudes * amplitudes.conj()).real
        cross += self.coulomb_cross_section(angles)

        return angles, cross
    
    def partial_wave_amplitude(self, angles: numpy.ndarray, l: int, rmax: int, dr: float) -> numpy.ndarray:
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
        centrifugal_potential = self.partial_wave_potential(l)
        solutions = self.radial_solutions(centrifugal_potential, 2.0 * l * dr, rmax, dr)

        a = self.outward_radiuses()
        smatrix = self.smatrix(a, l, solutions)

        radians = angles * numpy.pi / 180
        legendre = Legendre(l)
        coulomb_dl = arg_gamma(complex(l + 1, self.sommerfield))

        return 1 / (self.wavenumber) * (2 * l + 1) * numpy.exp(coulomb_dl) * legendre(numpy.cos(radians)) * (smatrix - 1)

    def partial_wave_potential(self, l: int):
        '''
        Method for creating full potential of partial-wave dispansion.

        Params
        ------
        `l` : `int`
            Partial wave number.

        Returns
        -------
        `potential` : `lambda`
            Partial wave potential.
        '''
        c = 3e23 # fm / s
        h_bar = 6.582119e-22 # MeV * s
        mu = self.reduced_mass
        ecm = self.center_mass_energy

        return lambda r: l * (l + 1) / (r ** 2) + 2 * mu / (h_bar ** 2 * c ** 2) * (ecm - self.potential(r))
    
    def radial_solutions(self, potential, rmin: float, rmax: float, dr: float) -> tuple[numpy.ndarray, numpy.ndarray]:
        '''
        Method for numerical solving Schrodinger equations with given potential `self.potential`

        Params
        ------
        `rmax` : `float`
            Maximal value of radius for integrating.\n

        `dr` : `float`
            Integration step.

        Returns
        --------
        `radials` : `tuple[numpy.ndarray[complex], numpy.ndarray[complex]]`
            Solution of Schrodinger eq. 
        '''
        return Numerov(potential, rmin, rmax, dr).solve()

    def outward_radiuses(self) -> float:
        '''
        Method that \'calculates\' outward radiuses `a`.

        Returns
        -------
        `radius` : `float`
            Matching outward radius in fermi.
        '''
        return 4 * numpy.pi * self.wavenumber
    
    def smatrix(self, a: float, l: int, solutions: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> complex:
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
        `dl` : `complex`
            Phase shift of certain partial wave, dimensionless.
        '''
        index = numpy.abs(solutions[0] - a).argmin()
        xl1 = solutions[1][index]
        dxl1 = (solutions[1][index + 1] + solutions[1][index - 1]) / (2 * (solutions[0][index] - solutions[0][index - 1]))

        hminus = CoulombWaveFunction(l, False)
        hplus = CoulombWaveFunction(l, True)

        etha = self.sommerfield
        rmatrix = 1 / a * (xl1 / dxl1)
        numerator = hminus(etha, self.wavenumber * a) - a * rmatrix * hminus.derivative(etha, self.wavenumber * a)
        denumerator = hplus(etha, self.wavenumber * a) - a * rmatrix * hplus.derivative(etha, self.wavenumber * a)

        smatrix = numerator / denumerator

        return complex(smatrix)
    
    def coulomb_cross_section(self, thetas: numpy.ndarray) -> numpy.ndarray:
        '''
        Params
        ------
        `thetas` : `numpy.ndarray`
            Angles coulomb amplitude calculates to, deg.

        Returns
        -------
        `fc` : `numpy.ndarray[float]`
            Coulomb scattering amplitude, (mb/sr)^(1/2)
        '''
        radians = thetas * numpy.pi / 180
        reduced_planck = 6.582e-22 # MeV * s
        lightspeed = 3e23 # fm / s
        fine_structure = 1 / 137 # dimensionless

        energy_cm = self._energy * (1 - self._beam.mass() / (self._beam.mass() + self._target.mass())) # MeV
        e_power_2 = fine_structure * reduced_planck * lightspeed # MeV * fm
        numerator = self._beam.charge * self._target.charge * e_power_2 # MeV * fm
        denumerator = 4 * energy_cm * numpy.sin(radians / 2) ** 2 # MeV * rad

        return numpy.power(numerator / denumerator, 2) * 10 # mb/sr

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

    Vd = 99.03; rv = 1.20; av = 0.755
    Ws = 20.96; rw = 1.31; aw = 0.645
    rc = 1.28

    opt = Optical(beam, target)
    opt.add_real_volume(Vd, rv, av)
    opt.add_imag_surface(Ws, rw, aw)
    opt.add_coulomb(rc)

    elastic = Elastic(opt, E_lab)
    angles, cross = elastic.xsections(1, 180, 0.5)

    exp_ang, exp_xs = [], []
    with open('src/exp.txt', 'r') as file:
        buffer = file.read().split('\n')
        for line in buffer:
            exp_ang.append(float(line.split()[0]))
            exp_xs.append(float(line.split()[1]))

    plt.plot(angles, cross, color='blue')
    plt.scatter(exp_ang, exp_xs, color='black')
    plt.yscale('log')
    plt.grid()
    plt.show()