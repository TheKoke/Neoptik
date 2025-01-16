from __future__ import annotations
from abc import ABC, abstractmethod

import numpy
from nuclear import Nuclei
from mathematics.integration import Simpson


class Potential(ABC):
    def __init__(self, beam: Nuclei, target: Nuclei) -> None:
        self._beam = beam
        self._target = target

    @property
    def beam(self) -> Nuclei:
        return self._beam
    
    @property
    def target(self) -> Nuclei:
        return self._target
    
    def is_beam_negligible(self) -> bool:
        return self._beam.nuclons <= 4

    @abstractmethod
    def function(self, r: float) -> float:
        pass

    def volume_integral(self) -> float:
        """
        The volume integral of potential - `J`

        Returns
        -------
        Volume integral of potential - `Jv`. MeV * fm^3
        """
        return Simpson.take_integral(self.function, 0.0, 30.0) / (self._beam.nuclons * self._target.nuclons)


class ZeroPotential(Potential):
    def __init__(self):
        super().__init__(None, None)

    def function(self, r: float):
        return 0.0


class Coulomb(Potential):
    def __init__(self, beam: Nuclei, target: Nuclei, Rc: float) -> None:
        """
        Couloumb potential near nuclear radiuses.

        Parameters
        ----------
        `beam` : `Nuclei`
            beam nucleus.

        `target` : `Nuclei`
            Target nucleus.

        `Rc` : float 
            Couloumb barrier radius, fm.
        """
        super().__init__(beam, target)

        self._z1 = self._beam.charge
        self._z2 = self._target.charge
        self._rc = Rc

    @property
    def z1(self) -> int:
        return self._z1
    
    @property
    def z2(self) -> int:
        return self._z2
    
    @property
    def Rc(self) -> float:
        return self._rc

    def function(self, r: float) -> float:
        """
        Coulomb potential near nuclear radiuses - `Vc(r)`.

        Parameters
        ----------
        `r` : float
            Radius-vector, fm.

        Returns
        -------
        `Vc` : float
            Value of coulomb potential at point `r`, MeV.
        """
        reduced_planck = 6.582e-22 # MeV * s
        lightspeed = 3e23 # fm / s
        fine_structure = 1 / 137 # dimensionless
        e2 = fine_structure * reduced_planck * lightspeed # MeV * fm
        rc = self._rc * numpy.cbrt(self._target.nuclons) if self.is_beam_negligible() \
            else self._rc * (numpy.cbrt(self._target.nuclons) + numpy.cbrt(self._beam.nuclons))

        return -self._z1 * self._z2 * e2 / rc * (3 / 2 - r ** 2 / (2 * rc ** 2)) if r <= rc \
                else -self._z1 * self._z2 * e2 / r # MeV



class WSParameters:
    def __init__(self, V: float, R: float, a: float) -> None:
        """
        Parameters of Woods-Saxon potenial form-factor.

        Parameters
        ----------
        `V` : float
            Depth of potential, MeV.

        `R` : float
            Radius or distance of potential action, fm.

        `a` : float
            Diffuseness of nuclear-shell, fm.
        """
        self._V = V
        self._R = R
        self._a = a

    @property
    def V(self) -> float:
        return self._V
    
    @property
    def R(self) -> float:
        return self._R
    
    @property
    def a(self) -> float:
        return self._a


class WSVolume(Potential):
    def __init__(self, beam: Nuclei, target: Nuclei, params: WSParameters, is_imag: bool = False) -> None:
        """
        Woods-Saxon Volume potential.

        Parameters
        ----------
        `beam` : `Nuclei`
            beam nucleus.

        `target` : `Nuclei`
            Target nucleus.

        `params` : `WSParameters`
            Parameters of potential.

        `is_imag` : `bool`
            Flag for imaginary potential, if is imaginary it is equal True, otherwaise False.
        """
        super().__init__(beam, target)

        self._V = params.V
        self._R = params.R
        self._a = params.a

        self._is_imag = is_imag

    @property
    def V(self) -> float:
        return self._V
    
    @property
    def R(self) -> float:
        return self._R

    @property
    def a(self) -> float:
        return self._a
    
    def function(self, r: float) -> float:
        """
        Woods-Saxon Volume potential - `WS_V(r)`.

        Parameters
        ----------
        `r` : float
            Radius-vector, fm.

        Returns
        -------
        `WS` : float
            Value of Woods-Saxon potential at point `r`, MeV.
        """
        r_int = self._R * (numpy.cbrt(self._target.nuclons)) if self.is_beam_negligible() \
            else self._R * (numpy.cbrt(self._target.nuclons) + numpy.cbrt(self._beam.nuclons))

        value = -self._V / (1 + numpy.exp((r - r_int) / self._a))
        return value if not self._is_imag else 1j * value


class WSSurface(Potential):
    def __init__(self, beam: Nuclei, target: Nuclei, params: WSParameters, is_imag: bool = False) -> None:
        """
        Woods-Saxon Surface potential.

        Parameters
        ----------
        `beam` : `Nuclei`
            beam nucleus.

        `target` : `Nuclei`
            Target nucleus.

        `params` : `WSParameters`
            Parameters of potential.

        `is_imag` : `bool`
            Flag for imaginary potential, if is imaginary it is equal True, otherwaise False.
        """
        super().__init__(beam, target)

        self._V = params.V
        self._R = params.R
        self._a = params.a

        self._is_imag = is_imag

    @property
    def V(self) -> float:
        return self._V
    
    @property
    def R(self) -> float:
        return self._R

    @property
    def a(self) -> float:
        return self._a
    
    def function(self, r: float) -> float:
        """
        Woods-Saxon Surface potential - `WS_D(r)`.

        Parameters
        ----------
        `r` : float
            Radius-vector, fm.

        Returns
        -------
        `WS` : float
            Value of Woods-Saxon potential at point `r`, MeV.
        """
        r_int = self._R * (numpy.cbrt(self._target.nuclons)) if self.is_beam_negligible() \
            else self._R * (numpy.cbrt(self._target.nuclons) + numpy.cbrt(self._beam.nuclons))

        value = -4 * self._V * numpy.exp((r - r_int) / self._a) / (1 + numpy.exp((r - r_int) / self._a)) ** 2
        return value if not self._is_imag else 1j * value


class SpinOrbit(Potential):
    def __init__(self, beam: Nuclei, target: Nuclei, real_params: WSParameters, imag_params: WSParameters) -> None:
        """
        Spin-Orbital interaction potential.

        Parameters
        ----------
        `beam` : `Nuclei`
            beam nucleus.

        `target` : `Nuclei`
            Target nucleus.

        `real_params` : `WSParameters`
            Parameters of real_part of potential.

        `imag_params` : `WSParameters`
            Parameters of imaginary part of potential.
        """
        super().__init__(beam, target)

        self._V = real_params.V
        self._r = real_params.R
        self._a = real_params.a
        self._W = imag_params.V
        self._ir = imag_params.R
        self._ia = imag_params.a

    @property
    def V(self) -> float:
        return self._V
    
    @property
    def R(self) -> float:
        return self._r
    
    @property
    def a(self) -> float:
        return self._a
    
    @property
    def W(self) -> float:
        return self._W
    
    @property
    def iR(self) -> float:
        return self._ir
    
    @property
    def ia(self) -> float:
        return self._ia
    
    def function(self, r: float) -> complex:
        """
        Spin-Orbit interaction potential - `SO(r)`

        Parameters
        ----------
        `r` : float
            Radius-vector, fm.

        Returns
        -------
        `SO` : float
            Value of Spin-Orbit interaction at point `r`, MeV.
        """

        reduced_planck = 6.582e-22 # MeV * s
        lightspeed = 3e23 # fm / s
        pion_mass = 134.977 # MeV
        coefficient = (reduced_planck * lightspeed / pion_mass) ** 2 # fm^2

        r_int = self._r * (numpy.cbrt(self._target.nuclons)) if self.is_beam_negligible() \
            else self._r * (numpy.cbrt(self._target.nuclons) + numpy.cbrt(self._beam.nuclons))
        
        ir_int = self._ir * (numpy.cbrt(self._target.nuclons)) if self.is_beam_negligible() \
            else self._ir * (numpy.cbrt(self._target.nuclons) + numpy.cbrt(self._beam.nuclons))
        
        real_exp = numpy.exp((r - r_int) / self._a)
        imag_exp = numpy.exp((r - ir_int) / self._a)

        real_part = self._V * 1 / (r * self._a) * real_exp / (1 + real_exp) ** 2 # MeV / fm
        imag_part = self._W * 1 / (r * self._ia) * imag_exp / (1 + imag_exp) ** 2 # MeV / fm

        # Add scalar multiplication of l and s.
        # (l, s) => 1/2 * [j(j + 1) - l(l + 1) - s(s + 1)]
        # add parameters of incident particle: l: int, s: float.

        return coefficient * (real_part + 1j * imag_part)
    

class Optical:
    def __init__(self, beam: Nuclei, target: Nuclei, 
                 real: Potential, imag: Potential,
                 coul: Potential = ZeroPotential(), 
                 spin: Potential = ZeroPotential()) -> None:
        self._beam = beam
        self._target = target

        self._coulomb = coul
        self._real_part = real
        self._imag_part = imag
        self._spinorbit = spin

    def __call__(self, r: float) -> float:
        return self._real_part.function(r) + \
               self._imag_part.function(r) + \
               self._coulomb.function(r) + \
               self._spinorbit.function(r)

    @property
    def beam(self) -> Nuclei:
        return self._beam
    
    @property
    def target(self) -> Nuclei:
        return self._target
    
    @property
    def real_part(self) -> Potential:
        return self._real_part
    
    @property
    def imaginary_part(self) -> Potential:
        return self._imag_part
    
    @property
    def coulomb_potential(self) -> Potential:
        return self._coulomb
    
    @property
    def spin_orbit(self) -> Potential:
        return self._spinorbit


if __name__ == '__main__':
    pass
