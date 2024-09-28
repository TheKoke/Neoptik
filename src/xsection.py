import numpy
import matplotlib.pyplot as plt


from chi import Chi_Square
from nuclear import Nuclei
from numerov import Numerov
from legendre import Legendre
from potentials import Optical
from bessel import SphericalBesselJ, SphericalBesselY


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
                  lmax: int = 10, 
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
            Count of partial waves for sum, encounts from 0. Default value = 10.

        `rmax` : `float`
            Maximal radius for integration, fm. Default value = 20.0.

        `dr` : `float`
            Integration step size, fm. Default value = 0.01.

        Returns
        -------

        `angles, xsections` : `tuple[numpy.ndarray, numpy.ndarray]`
            Elastic cross-section. Angles in degrees, xsections in mb/sr.
        '''

        angles = numpy.linspace(theta0, thetan, int((thetan - theta0) / dtheta))
        cross = numpy.zeros_like(angles)

        for i in range(lmax + 1):
            effective_potential = self.partial_wave_potential(i)
            
            solutions = self.radial_solutions(effective_potential, rmax, dr)
            r1, r2 = self.outward_radiuses()

            plt.plot(solutions[0], solutions[1])
            plt.show()

            phase_shift = self.phase_shift(r1, r2, solutions, i)
            amplitude = self.scattering_amplitude(angles, phase_shift, i)
            cross += (amplitude * amplitude.conjugate()).real

        return angles, cross

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

        return lambda r: l * (l + 1) / r ** 2 + 2 * mu / (h_bar ** 2 * c ** 2) * (center_mass_energy - self._potential(r))
    
    def radial_solutions(self, potential, rmax: float, dr: float) -> tuple[numpy.ndarray, numpy.ndarray]:
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
        `radials` : `tuple[numpy.ndarray, numpy.ndarray]`
            Solution of Schrodinger eq. 
        '''
        return Numerov(potential, rmax, dr).solve()

    def outward_radiuses(self) -> tuple[float, float]:
        '''
        Method that \'calculates\' outward radiuses `r1` and `r2`

        Returns
        -------
        `radiuses` : `tuple[float, float]`
            Matching outward radiuses in fermi.
        '''
        return (20, 25) # TODO: brute-force, fix and try to automatize.

    def phase_shift(self, r1: float, r2: float, solutions: tuple[numpy.ndarray, numpy.ndarray], l: int) -> float:
        '''
        Params
        ------
        `r1` : `float`
            First outward radius, fm.

        `r2` : `float`
            Second outward radius, fm.

        `solutions` : `tuple[numpy.ndarray, numpy.ndarray]`
            Solution of Schrodinger equation.

        `l` : `int`
            Partial wave number.

        Returns
        -------
        `phase_shift` : `float`
            Phase shift of outgoing wave, dimensionless.
        '''
        xl1 = solutions[1][(numpy.abs(solutions[0] - r1)).argmin()]
        xl2 = solutions[1][(numpy.abs(solutions[0] - r2)).argmin()]
        kmatrix = (r2 * xl1) / (r1 * xl2)

        jl = SphericalBesselJ(l)
        nl = SphericalBesselY(l)

        numerator = kmatrix * jl(self.wavenumber * r2) - jl(self.wavenumber * r1)
        denumerator = kmatrix * nl(self.wavenumber * r2) - nl(self.wavenumber * r1)

        return numpy.arctan(numerator / denumerator)
    
    def scattering_amplitude(self, thetas: numpy.ndarray, phase_shift: float, l: int) -> complex:
        '''
        Params
        ------
        `thetas` : `numpy.ndarray`
            Angles to calculate amplitude, deg.

        `phase_shift` : `complex`
            Phase shift of outgoing wave, dimensionless.

        `l` : `int`'
            Partial-wave number.

        Returns
        -------
        `f` : `complex`
            Scattering amplitude of wave.
        '''
        radians = thetas * numpy.pi / 180
        legendre = Legendre(l)

        return 1 / self.wavenumber * (2 * l + 1) * legendre(numpy.cos(radians)) \
                * numpy.exp(complex(0, phase_shift)) * numpy.sin(phase_shift)

    def chi_square(self, theory: numpy.ndarray, experimenthal: numpy.ndarray, uncertainty: numpy.ndarray) -> float:
        '''
        Params
        ------
        `theory` : `numpy.ndarray`
            Theoretical calculated cross-sections, mb/sr.

        `experimenthal` : `numpy.ndarray`
            Experimenthal measured cross-sections, mb/sr.
        
        `uncertainty` : `numpy.ndarray`
            Uncertatinty of experimenthal cross-sections, dimensionless.

        Returns
        -------
        `chi-square` : `float`
            Chi-square for calculation.
        '''
        return Chi_Square.averaged_chi_square(theory, experimenthal, uncertainty)


if __name__ == '__main__':
    beam = Nuclei(1, 2)
    target = Nuclei(6, 13)
    E_lab = 18

    Vd = 99.03; rv = 1.20; av = 0.755
    Ws = 20.96; rw = 1.31; aw = 0.645

    opt = Optical(beam, target)
    opt.add_real_volume(Vd, rv, av)
    opt.add_imag_surface(Ws, rw, aw)
    opt.add_coulomb(1.28)

    elastic = Elastic(opt, E_lab)
    
    angles, cross = elastic.xsections(2, 180, 2)

    plt.plot(angles, cross)
    plt.yscale('log')
    plt.grid()
    plt.show()