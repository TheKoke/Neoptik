from __future__ import annotations
import os


NAME2CHARGE = {
    'p' :   1, 'd' :   1, 't' :   1,
    'h' :   1, 'he':   2, 'li':   3, 'be':   4, 'b' :   5, 'c' :   6,
    'n' :   7, 'o' :   8, 'f' :   9, 'ne':  10, 'na':  11, 'mg':  12,
    'al':  13, 'si':  14, 'p' :  15, 's' :  16, 'cl':  17, 'ar':  18,
    'k' :  19, 'ca':  20, 'sc':  21, 'ti':  22, 'v' :  23, 'cr':  24,
    'mn':  25, 'fe':  26, 'co':  27, 'ni':  28, 'cu':  29, 'zr':  30,
    'ga':  31, 'ge':  32, 'as':  33, 'se':  34, 'br':  35, 'kr':  36,
    'rb':  37, 'sr':  38, 'y' :  39, 'zr':  40, 'nb':  41, 'mo':  42,
    'tc':  43, 'ru':  44, 'rh':  45, 'pd':  46, 'ag':  47, 'cd':  48,
    'in':  49, 'sn':  50, 'sb':  51, 'te':  52, 'i' :  53, 'xe':  54,
    'cs':  55, 'ba':  56, 'la':  57, 'ce':  58, 'pr':  59, 'nd':  60,
    'pm':  61, 'sm':  62, 'eu':  63, 'gd':  64, 'tb':  65, 'dy':  66,
    'ho':  67, 'er':  68, 'tm':  69, 'yb':  70, 'lu':  71, 'hf':  72,
    'ta':  73, 'w' :  74, 're':  75, 'os':  76, 'ir':  77, 'pt':  78,
    'au':  79, 'hg':  80, 'tl':  81, 'pb':  82, 'bi':  83, 'po':  84,
    'at':  85, 'rn':  86, 'fr':  87, 'ra':  88, 'ac':  89, 'th':  90,
    'pa':  91, 'u' :  92, 'np':  93, 'pu':  94, 'am':  95, 'cm':  96,
    'bk':  97, 'cf':  98, 'es':  99, 'fm': 100, 'md': 101, 'no': 102,
    'lr': 103, 'rf': 104, 'db': 105, 'sg': 106, 'bh': 107, 'hs': 108,
    'mt': 109, 'ds': 110, 'rg': 111, 'cn': 112, 'nh': 113, 'fl': 114,
    'mc': 115, 'lv': 116, 'ts': 117, 'og': 118
}


CHARGE2NAME = {
    0  :  'n',
    1  :  'H', 2  : 'He', 3  : 'Li', 4  : 'Be', 5  :  'B', 6  :  'C',
    7  :  'N', 8  :  'O', 9  :  'F', 10 : 'Ne', 11 : 'Na', 12 : 'Mg',
    13 : 'Al', 14 : 'Si', 15 :  'P', 16 :  'S', 17 : 'Cl', 18 : 'Ar',
    19 :  'K', 20 : 'Ca', 21 : 'Sc', 22 : 'Ti', 23 :  'V', 24 : 'Cr',
    25 : 'Mn', 26 : 'Fe', 27 : 'Co', 28 : 'Ni', 29 : 'Cu', 30 : 'Zr',
    31 : 'Ga', 32 : 'Ge', 33 : 'As', 34 : 'Se', 35 : 'Br', 36 : 'Kr',
    37 : 'Rb', 38 : 'Sr', 39 :  'Y', 40 : 'Zr', 41 : 'Nb', 42 : 'Mo',
    43 : 'Tc', 44 : 'Ru', 45 : 'Rh', 46 : 'Pd', 47 : 'Ag', 48 : 'Cd',
    49 : 'In', 50 : 'Sn', 51 : 'Sb', 52 : 'Te', 53 :  'I', 54 : 'Xe',
    55 : 'Cs', 56 : 'Ba', 57 : 'La', 58 : 'Ce', 59 : 'Pr', 60 : 'Nd',
    61 : 'Pm', 62 : 'Sm', 63 : 'Eu', 64 : 'Gd', 65 : 'Tb', 66 : 'Dy',
    67 : 'Ho', 68 : 'Er', 69 : 'Tm', 70 : 'Yb', 71 : 'Lu', 72 : 'Hf',
    73 : 'Ta', 74 :  'W', 75 : 'Re', 76 : 'Os', 77 : 'Ir', 78 : 'Pt',
    79 : 'Au', 80 : 'Hg', 81 : 'Tl', 82 : 'Pb', 83 : 'Bi', 84 : 'Po',
    85 : 'At', 86 : 'Rn', 87 : 'Fr', 88 : 'Ra', 89 : 'Ac', 90 : 'Th',
    91 : 'Pa', 92 :  'U', 93 : 'Np', 94 : 'Pu', 95 : 'Am', 96 : 'Cm',
    97 : 'Bk', 98 : 'Cf', 99 : 'Es', 100: 'Fm', 101: 'Md', 102: 'No',
    103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh', 108: 'Hs',
    109: 'Mt', 110: 'Ds', 111: 'Rg', 112: 'Cn', 113: 'Nh', 114: 'Fl',
    115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og'
}


INDENT = 3
NUCLEI_SIZE = 5
ENERGY_SIZE = 10
PARITY_SIZE = 10
GAMMA_SIZE = 10


def mass_excess_of(z: int, a: int) -> float:
    file = open(find_file(z), 'r').read().split('\n')
    start, _ = find_nuclei_area(z, a, file)

    return get_mass_excess(file[start])

def excitation_energies(z: int, a: int) -> list[float]:
    file = open(find_file(z), 'r').read().split('\n')
    start, stop = find_nuclei_area(z, a, file)

    collected = []
    for i in range(start + 1, stop):
        collected.append(get_energy(file[i]))

    return collected

def spin_parity(z: int, a: int) -> list[tuple[float, bool]]:
    file = open(find_file(z), 'r').read().split('\n')
    start, stop = find_nuclei_area(z, a, file)

    collected = []
    for i in range(start + 1, stop):
        collected.append(get_spin_parity(file[i]))

    return collected

def gammas(z: int, a: int) -> list[float]:
    file = open(find_file(z), 'r').read().split('\n')
    start, stop = find_nuclei_area(z, a, file)

    collected = []
    for i in range(start + 1, stop):
        collected.append(get_gamma(file[i]))

    return collected


def find_file(z: int) -> str:
    ensdf_path = os.getcwd() + '\\src\\ensdf\\'
    files = os.listdir(ensdf_path)

    choosen = CHARGE2NAME[z]
    looking_file = f'{z}{choosen}.txt'

    if looking_file not in files:
        raise ValueError(f'Can not find the ensdf file for {choosen} nuclei')
    
    return ensdf_path + looking_file

def find_nuclei_area(z: int, a: int, buffer: list[str]) -> tuple[int, int]:
    choosen = CHARGE2NAME[z].upper()
    looking_line = f'{a}{choosen}'

    start, stop = 0, 0

    is_opened = False
    for i in range(len(buffer)):
        if looking_line in buffer[i] and not is_opened:
            start = i
            is_opened = True

        if looking_line not in buffer[i] and is_opened:
            stop = i
            return (start, stop)

    if not is_opened:
        raise ValueError(f'Cannot find nuclei with z: {z} and a: {a}')

def get_mass_excess(line: str) -> float:
    mass_excess_flag = 'deltaM='
    if mass_excess_flag not in line:
        return 0
    
    start_index = line.index(mass_excess_flag) + len(mass_excess_flag)
    visible = ''
    for i in range(start_index, len(line)):
        if line[i].isdigit() or line[i] == '.' or line[i] == '-':
            visible += line[i]
        else:
            break

    return float(visible)

def get_energy(line: str) -> float:
    offset = INDENT + NUCLEI_SIZE
    area = line[offset: offset + ENERGY_SIZE]
    
    numbers = ''.join([i for i in area if i.isdigit() or i == '.'])
    return float(numbers)

def get_spin_parity(line: str) -> tuple[float, bool]:
    offset = 2 * INDENT + NUCLEI_SIZE + ENERGY_SIZE
    area = line[offset: offset + PARITY_SIZE]

    spin = ''.join([i for i in area if i.isdigit() or i == '.'])
    parity = ''.join([i for i in area if i == '+' or i == '-'])

    return (float(spin), parity == '+')

def get_gamma(line: str) -> float:
    offset = 3 * INDENT + NUCLEI_SIZE + ENERGY_SIZE + PARITY_SIZE
    area = line[offset: offset + GAMMA_SIZE]

    unit_index = area.index(' ')
    return float(area[:unit_index])


if __name__ == '__main__':
    pass
