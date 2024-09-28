from __future__ import annotations
from informer import Informator, NAME2CHARGE


class Nuclei:
    def __init__(self, charge: int, nuclons: int) -> None:
        if not Informator.is_exist(charge, nuclons):
            raise ValueError(f'Not such a nuclei with z={charge} and a={nuclons}')

        self.nuclons = nuclons
        self.charge = charge
        self.name = Informator.name(self.charge, self.nuclons)

    @property
    def mass_excess(self) -> float:
        return Informator.mass_excess(self.charge, self.nuclons)
    
    @property
    def states(self) -> list[float]:
        return Informator.states(self.charge, self.nuclons)
    
    @property
    def spins(self) -> list[tuple[float, bool]]:
        return Informator.spins(self.charge, self.nuclons)
    
    @property
    def wigner_widths(self) -> list[float]:
        return Informator.wigner_widths(self.charge, self.nuclons)
    
    @property
    def spin(self) -> float:
        return self.spins[0][0]
    
    @property
    def radius(self) -> float:
        fermi = 1.28 # fm
        return fermi * (self.nuclons) ** 1 / 3 # fm
    
    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return self.name
    
    def __str__(self) -> str:
        return self.name
    
    def __eq__(self, other: Nuclei) -> bool:
        return self.nuclons == other.nuclons and self.charge == other.charge
    
    def __add__(self, other: Nuclei) -> Nuclei:
        return Nuclei(self.charge + other.charge, self.nuclons + other.nuclons)
    
    def __sub__(self, other: Nuclei) -> Nuclei:
        return Nuclei(self.charge - other.charge, self.nuclons - other.nuclons)
    
    def mass(self, unit: str = 'MeV') -> float:
        match unit.lower():
            case 'mev': 
                return self.charge * 938.27 + (self.nuclons - self.charge) * 939.57
            case 'amu' | 'a.m.u':
                return self.charge * 1.0073 + (self.nuclons - self.charge) * 1.0087
            case 'g':
                return self.charge * 1.672e-24 + (self.nuclons - self.charge) * 1.675e-24
            case _:
                return self.nuclons
            
    @staticmethod
    def from_string(input: str) -> Nuclei:
        if Nuclei.handle_human_namings(input) is not None:
            return Nuclei.handle_human_namings(input)

        only_name = ''.join([i.lower() for i in input if i.isalpha()])
        only_nuclons = ''.join([i for i in input if i.isdigit()])

        charge = NAME2CHARGE[only_name]
        nuclon = int(only_nuclons)
        
        return Nuclei(charge, nuclon)
    
    @staticmethod
    def handle_human_namings(input: str) -> Nuclei:
        if input in ['p', 'd', 't']:
            return Nuclei(1, ['p', 'd', 't'].index(input) + 1)
        
        if input == 'alpha' or input == 'α' or input == 'a':
            return Nuclei(2, 4)
        
        return None


if __name__ == '__main__':
    pass
