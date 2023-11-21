def get_mass_from_atomic_number(atomic_number):
    # NOTE: there might be a better way to do this
    # define a dictionary mapping atomic numbers to masses
    atomic_masses = {
        1: 1.008,
        6: 12.011,
        8: 15.999,
        # add more atomic numbers and corresponding masses as needed
    }

    # check if the atomic number is in the dictionary
    if atomic_number in atomic_masses:
        return atomic_masses[atomic_number]
    else:
        raise KeyError(f"Atomic number {atomic_number} not found.")


def get_symbol_from_atomic_number(atomic_number):
    # define a dictionary mapping atomic numbers to chemical symbols
    atomic_symbols = {
        1: "H",
        6: "C",
        8: "O",
        # add more atomic numbers and corresponding symbols as needed
    }

    # check if the atomic number is in the dictionary
    if atomic_number in atomic_symbols:
        return atomic_symbols[atomic_number]
    else:
        raise KeyError(f"Atomic number {atomic_number} not found.")
