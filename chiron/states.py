from openmm import unit


class StateVariablesCollection:
    """
    Defines all variables that are necessary to construct a joint distribution of
    individual conditional distributions that should be sampled.
    Keeps these variables updated as they change during simulation

    """

    def __init__(self) -> None:
        # initialize all state variables
        self.temperature: unit.Quantity
        self.volumn: unit.Quantity
        self.pressure: unit.Quantity
        self.nr_of_particles: int
        self.position: unit.Quantity
        

    @property
    def kb(self):
        return unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

    @property
    def beta(self):
        """Thermodynamic beta in units of mole/energy."""
        return 1.0 / (self.kb * self.temperature)

