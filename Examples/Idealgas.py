from openmmtools.testsystems import IdealGas
from openmm import unit

"""
This example explore an ideal gas system, where the particles are non-interacting. 
This will use the MonteCarloBarostatMove to sample the volume of the system and 
MetropolisDisplacementMove to sample the particle positions.

This utilizes the IdealGas example from openmmtools to initialize particle positions and topology.

"""

# Use the IdealGas example from openmmtools to initialize particle positions and topology
# For this example, the topology provides the masses for the particles

n_particles = 216
temperature = 298 * unit.kelvin
pressure = 1 * unit.atmosphere

ideal_gas = IdealGas(nparticles=n_particles, temperature=temperature, pressure=pressure)


from chiron.potential import IdealGasPotential
from chiron.utils import PRNG, get_list_of_mass
import jax.numpy as jnp

# particles are non interacting
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
# but will be used to appropriately wrap particles in space
nbr_list = PairList(OrthogonalPeriodicSpace(), cutoff=cutoff)
nbr_list.build_from_state(sampler_state)

from chiron.reporters import MCReporter

# initialize a reporter to save the simulation data
filename = "test_mc_ideal_gas.h5"
import os

if os.path.isfile(filename):
    os.remove(filename)
reporter = MCReporter(filename, 100)


from chiron.mcmc import (
    MetropolisDisplacementMove,
    MonteCarloBarostatMove,
    MoveSchedule,
    MCMCSampler,
)

# initialize the displacement move
mc_barostat_move = MonteCarloBarostatMove(
    volume_max_scale=0.2,
    number_of_moves=10,
    reporter=reporter,
    autotune=True,
    autotune_interval=100,
)

# initialize the barostat move and the move schedule
metropolis_displacement_move = MetropolisDisplacementMove(
    displacement_sigma=0.1 * unit.nanometer,
    number_of_moves=100,
    autotune=True,
    autotune_interval=100,
)

# define the move schedule
move_set = MoveSchedule(
    [
        ("MetropolisDisplacementMove", metropolis_displacement_move),
        ("MonteCarloBarostatMove", mc_barostat_move),
    ]
)

sampler = MCMCSampler(move_set)
sampler.run(
    sampler_state, thermodynamic_state, n_iterations=10, nbr_list=nbr_list
)  # how many times to repeat

# get the volume from the reporter
volume = reporter.get_property("volume")
step = reporter.get_property("elapsed_step")


import matplotlib.pyplot as plt

plt.plot(step, volume)
plt.show()

# get expectations
ideal_volume = ideal_gas.get_volume_expectation(thermodynamic_state)
ideal_volume_std = ideal_gas.get_volume_standard_deviation(thermodynamic_state)

print("ideal volume and standard deviation: ", ideal_volume, ideal_volume_std)


volume_mean = jnp.mean(jnp.array(volume)) * unit.nanometer**3
volume_std = jnp.std(jnp.array(volume)) * unit.nanometer**3


print("measured volume and standard deviation: ", volume_mean, volume_std)

# get the masses of particles from the topology
masses = get_list_of_mass(ideal_gas.topology)

sum_of_masses = jnp.sum(jnp.array(masses.value_in_unit(unit.amu))) * unit.amu

ideal_density = sum_of_masses / unit.AVOGADRO_CONSTANT_NA / ideal_volume
measured_density = sum_of_masses / unit.AVOGADRO_CONSTANT_NA / volume_mean

assert jnp.isclose(
    ideal_density.value_in_unit(unit.kilogram / unit.meter**3),
    measured_density.value_in_unit(unit.kilogram / unit.meter**3),
    atol=1e-1,
)
# see if within 5% of ideal volume
assert (
    abs(ideal_volume - volume_mean) / ideal_volume < 0.05
), f"Warning: {abs(ideal_volume - volume_mean) / ideal_volume} exceeds the 5% threshold"

# see if within 10% of the ideal standard deviation of the volume
assert (
    abs(ideal_volume_std - volume_std) / ideal_volume_std < 0.1
), f"Warning: {abs(ideal_volume_std - volume_std) / ideal_volume_std} exceeds the 10% threshold"
