from __future__ import annotations
from abc import ABC, abstractmethod

import numpy
from nuclear import Nuclei


class Potential(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def function(self, r: float) -> float:
        pass


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
        super().__init__()
        
        self._beam = beam
        self._target = target

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

        return self._z1 * self._z2 * e2 / self._rc * (3 / 2 - r ** 2 / (2 * self._rc ** 2)) if r <= self._rc \
                else self._z1 * self._z2 * e2 / r # MeV



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
        super().__init__()
        
        self._beam = beam
        self._target = target

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
    
    def is_beam_negligible(self) -> bool:
        return self._beam.nuclons < 5
    
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

        value = self._V / (1 + numpy.exp((r - r_int) / self._a))
        return value if not self._is_imag else complex(0, value)
    
    def volume_integral(self) -> float:
        """
        The volume integral of potential - `J`

        Returns
        -------
        Volume integral of potential - `Jv`. MeV * fm^3
        """
        pass


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
        super().__init__()
        
        self._beam = beam
        self._target = target

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
    
    def is_beam_negligible(self) -> bool:
        return self._beam.nuclons < 5
    
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

        value = 4 * self._V * numpy.exp((r - r_int) / self._a) / (1 + numpy.exp((r - r_int) / self._a)) ** 2
        return value if not self._is_imag else complex(0, value)
    
    def volume_integral(self) -> float:
        """
        The volume integral of potential - `J`

        Returns
        -------
        Volume integral of potential - `Jv`. MeV * fm^3
        """
        pass


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
        super().__init__()
        
        self._beam = beam
        self._target = target

        self._V = real_params.V
        self._W = imag_params.V
        self._R = real_params.R
        self._a = real_params.a

    @property
    def V(self) -> float:
        return self._V
    
    @property
    def R(self) -> float:
        return self._R
    
    @property
    def a(self) -> float:
        return self._a
    
    @property
    def W(self) -> float:
        return self._W
    
    def is_beam_negligible(self) -> bool:
        return self._beam.nuclons < 5
    
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

        r_int = self._R * (numpy.cbrt(self._target.nuclons)) if self.is_beam_negligible() \
            else self._R * (numpy.cbrt(self._target.nuclons) + numpy.cbrt(self._beam.nuclons))
        
        ir_int = self._R * (numpy.cbrt(self._target.nuclons)) if self.is_beam_negligible() \
            else self._R * (numpy.cbrt(self._target.nuclons) + numpy.cbrt(self._beam.nuclons))
        
        real_exp = numpy.exp((r - r_int) / self._a)
        imag_exp = numpy.exp((r - ir_int) / self._a)

        real_part = self._V * 1 / (r * self._a) * real_exp / (1 + real_exp) ** 2 # MeV / fm
        imag_part = self._W * 1 / (r * self._a) * imag_exp / (1 + imag_exp) ** 2 # MeV / fm

        # Add scalar multiplication of l and s.
        # (l, s) => 1/2 * [j(j + 1) - l(l + 1) - s(s + 1)]
        # add parameters of incident particle: l: int, s: float.

        return coefficient * (real_part + 1j * imag_part)
    

class Optical:
    def __init__(self, beam: Nuclei, target: Nuclei) -> None:
        self._beam = beam
        self._target = target

        self._potentials: list[Potential] = []

    def __call__(self, r: float) -> float:
        return sum(-pot.function(r) for pot in self._potentials)

    @property
    def beam(self) -> Nuclei:
        return self._beam
    
    @property
    def target(self) -> Nuclei:
        return self._target

    def add_real_volume(self, V: float, r: float, a: float) -> bool:
        params = WSParameters(V, r, a)
        self._potentials.append(WSVolume(self._beam, self._target, params))

    def add_real_surface(self, V: float, r: float, a: float) -> bool:
        params = WSParameters(V, r, a)
        self._potentials.append(WSSurface(self._beam, self._target, params))

    def add_imag_volume(self, W: float, r: float, a: float) -> bool:
        params = WSParameters(W, r, a)
        self._potentials.append(WSVolume(self._beam, self._target, params, is_imag=True))

    def add_imag_surface(self, W: float, r: float, a: float) -> bool:
        params = WSParameters(W, r, a)
        self._potentials.append(WSSurface(self._beam, self._target, params, is_imag=True))

    def add_spin_orbit(self, V: float, W: float, r: float, a: float, s: float, i: int) -> bool:
        realparams = WSParameters(V, r, a)
        imagparams = WSParameters(W, r, a)
        self._potentials.append(SpinOrbit(self._beam, self._target, realparams, imagparams))

    def add_coulomb(self, rc: float) -> bool:
        self._potentials.append(Coulomb(self._beam, self._target, rc))


if __name__ == '__main__':
    beam = Nuclei(1, 2)
    target = Nuclei(1, 2)
    params = WSParameters(20.96, 1.31, 0.645)
    w = WSSurface(beam, target, params, is_imag=True)

    import matplotlib.pyplot as plt

    rs = numpy.linspace(0, 20, 150)
    values = [-w.function(r).imag for r in rs]

    plt.plot(rs, values)
    plt.show()
