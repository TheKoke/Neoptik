import numpy
import multiprocessing

from nuclear import Nuclei
from potentials import Optical
from mathematics.chi import Chi_Square
from mathematics.numerov import Numerov
from mathematics.legendre import Legendre
from mathematics.couloumb import CoulombWaveFunction


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
    def wavenumber(self) -> float:
        '''
        Returns
        -------
        `wavenumber` : `float`
            Wavenumber of system in c.m., fm^(-1).
        '''
        h_bar = 6.582119e-22 # MeV * s
        c = 3e23 # fm / s
        mu = self._beam.mass() * self._target.mass() / (self._beam.mass() + self._target.mass()) # MeV
        center_mass_energy = self._energy * (1 - self._beam.mass() / (self._beam.mass() + self._target.mass())) # MeV

        return numpy.sqrt(2 * mu * center_mass_energy / (h_bar ** 2 * c ** 2))
    
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
                  rmax: float = 25.0, dr: float = 0.01) -> tuple[numpy.ndarray, numpy.ndarray]:
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
            Maximal radius for integration, fm. Default value = 20.0.

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

        amplitudes += sum([fl for fl in results])
        cross = self.coulomb_cross_section(angles)
        cross += (amplitudes * amplitudes.conj()).real

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
        solutions = self.radial_solutions(centrifugal_potential, 2.0 * l * dr + 0.001, rmax, dr)

        r1, r2 = self.outward_radiuses()
        smatrix = self.smatrix(r1, r2, solutions, l)

        return self.scattering_amplitude(angles, smatrix, l)

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
        h_bar = 6.582119e-22 # MeV * s
        c = 3e23 # fm / s
        mu = self._beam.mass() * self._target.mass() / (self._beam.mass() + self._target.mass()) # MeV
        center_mass_energy = self._energy * (1 - self._beam.mass() / (self._beam.mass() + self._target.mass())) # MeV

        return lambda r: l * (l + 1) / (r ** 2) + 2 * mu / (h_bar ** 2 * c ** 2) * (self.potential(r) - center_mass_energy)
    
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

    def outward_radiuses(self) -> tuple[float, float]:
        '''
        Method that \'calculates\' outward radiuses `r1` and `r2`

        Returns
        -------
        `radiuses` : `tuple[float, float]`
            Matching outward radiuses in fermi.
        '''
        return (27, 30) # TODO: brute-force, fix and try to automatize.

    def smatrix(self, r1: float, r2: float, solutions: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray], l: int) -> complex:
        '''
        Params
        ------
        `r1` : `float`
            First outward radius, fm.

        `r2` : `float`
            Second outward radius, fm.

        `solutions` : `tuple[numpy.ndarray[complex], numpy.ndarray[complex]]`
            Solution of Schrodinger equation.

        `l` : `int`
            Partial wave number.

        Returns
        -------
        `dl` : `complex`
            Phase shift of certain partial wave, dimensionless.
        '''
        xl1 = solutions[1][(numpy.abs(solutions[0] - r1)).argmin()]
        xl2 = solutions[1][(numpy.abs(solutions[0] - r2)).argmin()]

        hminus = CoulombWaveFunction(l, False)
        hplus = CoulombWaveFunction(l, True)

        fine_structure = 1 / 137 # dimensionless
        mu = self._beam.mass() * self._target.mass() / (self._beam.mass() + self._target.mass()) # MeV
        energy_cm = self._energy * (1 - self._beam.mass() / (self._beam.mass() + self._target.mass())) # MeV

        sommerfield = self._beam.charge * self._target.charge * fine_structure * numpy.sqrt(mu / (2 * energy_cm))

        relation = xl1 / xl2

        numerator = relation * hminus(sommerfield, r2) - hminus(sommerfield, r1)
        denumerator = relation * hplus(sommerfield, r2) - hminus(sommerfield, r1)
        smatrix = numerator / denumerator

        return complex(smatrix)
    
    def scattering_amplitude(self, thetas: numpy.ndarray, smatrix: complex, l: int) -> complex:
        '''
        Params
        ------
        `thetas` : `numpy.ndarray[float]`
            Angles to calculate amplitude, deg.

        `phase_shift` : `complex`
            Phase shift of outgoing wave, dimensionless.

        `l` : `int`'
            Partial-wave number.

        Returns
        -------
        `f` : `numpy.ndarray[complex]`
            Scattering amplitude of cetain partial wave, (mb/sr)^(1/2).
        '''
        radians = thetas * numpy.pi / 180
        legendre = Legendre(l)

        return 1 / (self.wavenumber) * (2 * l + 1) * legendre(numpy.cos(radians)) * (smatrix - 1)
    
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
    E_lab = 18

    Vd = 99.03; rv = 1.20; av = 0.755
    Ws = 20.96; rw = 1.31; aw = 0.645
    rc = 1.28

    opt = Optical(beam, target)
    opt.add_real_volume(Vd, rv, av)
    opt.add_imag_surface(Ws, rw, aw)
    opt.add_coulomb(rc)

    elastic = Elastic(opt, E_lab)
    
    angles, cross = elastic.xsections(1, 180, 0.5)

    with open('src/exp.txt', 'r') as file:
        buffer = file.read().split('\n')

    exp_ang, exp_xs = [], []
    for line in buffer:
        exp_ang.append(float(line.split(' ')[0]))
        exp_xs.append(float(line.split(' ')[1]))
        
    with open('src/plot.txt', 'r') as file:
        buffer = file.read().split('\n')[:-1]

    thr_ang, thr_xs = [], []
    for line in buffer:
        blocks = line.split('\t')
        thr_ang.append(float(blocks[0]))
        thr_xs.append(float(blocks[1]))

    plt.plot(angles, cross, color='blue')
    plt.scatter(exp_ang, exp_xs, color='blue')
    plt.plot(thr_ang, thr_xs, color='red')
    plt.yscale('log')
    plt.grid()
    plt.show()