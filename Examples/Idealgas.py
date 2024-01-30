from openmmtools.testsystems import IdealGas
from openmm import unit


# Use the LennardJonesFluid example from openmmtools to initialize particle positions and topology
# For this example, the topology provides the masses for the particles
# The default LennardJonesFluid example considers the system to be Argon with 39.9 amu

n_particles = 216
temperature = 298 * unit.kelvin
pressure = 1 * unit.atmosphere
mass = unit.Quantity(39.9, unit.gram / unit.mole)

ideal_gas = IdealGas(nparticles=n_particles, temperature=temperature, pressure=pressure)

from chiron.potential import IdealGasPotential
from chiron.utils import PRNG
import jax.numpy as jnp

#
cutoff = 0.0 * unit.nanometer
ideal_gas_potential = IdealGasPotential(ideal_gas.topology)

from chiron.states import SamplerState, ThermodynamicState

# define the thermodynamic state
thermodynamic_state = ThermodynamicState(
    potential=ideal_gas_potential,
    temperature=temperature,
    pressure=pressure,
)

PRNG.set_seed(1234)


# define the sampler state
sampler_state = SamplerState(
    x0=ideal_gas.positions,
    current_PRNG_key=PRNG.get_random_key(),
    box_vectors=ideal_gas.system.getDefaultPeriodicBoxVectors(),
)

from chiron.neighbors import PairList, OrthogonalPeriodicSpace

# define the pair list for an orthogonal periodic space
# since particles are non-interacting, this will not really do much
# but will appropriately wrap particles in space
nbr_list = PairList(OrthogonalPeriodicSpace(), cutoff=cutoff)
nbr_list.build_from_state(sampler_state)

from chiron.reporters import _SimulationReporter

# initialize a reporter to save the simulation data
filename = "test_mc_ideal_gas.h5"
import os

if os.path.isfile(filename):
    os.remove(filename)
reporter = _SimulationReporter(filename, 1)


from chiron.mcmc import (
    MetropolisDisplacementMove,
    MonteCarloBarostatMove,
    MoveSchedule,
    MCMCSampler,
)


mc_barostat_move = MonteCarloBarostatMove(
    volume_max_scale=0.2,
    nr_of_moves=100,
    reporter=reporter,
    update_stepsize=True,
    update_stepsize_frequency=100,
)
move_set = MoveSchedule(
    [
        ("MonteCarloBarostatMove", mc_barostat_move),
    ]
)

sampler = MCMCSampler(move_set, sampler_state, thermodynamic_state)
sampler.run(n_iterations=50, nbr_list=nbr_list)  # how many times to repeat

import h5py

with h5py.File(filename, "r") as f:
    volume = f["volume"][:]
    steps = f["step"][:]

# get expectations
ideal_volume = ideal_gas.get_volume_expectation(thermodynamic_state)
ideal_volume_std = ideal_gas.get_volume_standard_deviation(thermodynamic_state)

print(ideal_volume, ideal_volume_std)


volume_mean = jnp.mean(jnp.array(volume)) * unit.nanometer**3
volume_std = jnp.std(jnp.array(volume)) * unit.nanometer**3


print(volume_mean, volume_std)

ideal_density = mass * n_particles / unit.AVOGADRO_CONSTANT_NA / ideal_volume
measured_density = mass * n_particles / unit.AVOGADRO_CONSTANT_NA / volume_mean

assert jnp.isclose(
    ideal_density.value_in_unit(unit.kilogram / unit.meter**3),
    measured_density.value_in_unit(unit.kilogram / unit.meter**3),
    atol=1e-1,
)
# see if within 5% of ideal volume
assert abs(ideal_volume - volume_mean) / ideal_volume < 0.05

# see if within 10% of the ideal standard deviation of the volume
assert abs(ideal_volume_std - volume_std) / ideal_volume_std < 0.1
