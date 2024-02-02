from openmm import unit
from openmm import app


n_particles = 1100
temperature = 140 * unit.kelvin
pressure = 13.00765 * unit.atmosphere
mass = unit.Quantity(16.04, unit.gram / unit.mole)

# create the topology
lj_topology = app.Topology()
element = app.Element(1000, "CH4", "CH4", mass)
chain = lj_topology.addChain()
for i in range(n_particles):
    residue = lj_topology.addResidue("CH4", chain)
    lj_topology.addAtom("CH4", element, residue)

import jax.numpy as jnp

# these were generated in Mbuild using fill_box which wraps packmol
# a minimum spacing of 0.4 nm was used during construction.

positions = jnp.load("Examples/methane_coords.npy") * unit.nanometer

box_vectors = jnp.array(
    [
        [4.275021399280942, 0.0, 0.0],
        [0.0, 4.275021399280942, 0.0],
        [0.0, 0.0, 4.275021399280942],
    ]
)

from chiron.potential import LJPotential
from chiron.utils import PRNG
import jax.numpy as jnp

#

# initialize the LennardJones potential for UA-TraPPE methane
#
sigma = 0.373 * unit.nanometer
epsilon = 0.2941 * unit.kilocalories_per_mole
cutoff = 1.4 * unit.nanometer

lj_potential = LJPotential(lj_topology, sigma=sigma, epsilon=epsilon, cutoff=cutoff)

from chiron.states import SamplerState, ThermodynamicState

# define the thermodynamic state
thermodynamic_state = ThermodynamicState(
    potential=lj_potential,
    temperature=temperature,
    pressure=pressure,
)

PRNG.set_seed(1234)


# define the sampler state
sampler_state = SamplerState(
    x0=positions,
    current_PRNG_key=PRNG.get_random_key(),
    box_vectors=box_vectors * unit.nanometer,
)


from chiron.neighbors import PairList, OrthogonalPeriodicSpace

# define the pair list for an orthogonal periodic space
# since particles are non-interacting, this will not really do much
# but will appropriately wrap particles in space
nbr_list = PairList(OrthogonalPeriodicSpace(), cutoff=cutoff)
nbr_list.build_from_state(sampler_state)

# CRI: minimizer is not working correctly on my mac
# from chiron.minimze import minimize_energy
#
# results = minimize_energy(
#     sampler_state.x0, lj_potential.compute_energy, nbr_list, maxiter=100
# )
#
# min_x = results.params
#
# sampler_state.x0 = min_x

from chiron.reporters import MCReporter, LangevinDynamicsReporter

# initialize a reporter to save the simulation data
filename_barostat = "test_mc_lj_barostat.h5"
filename_displacement = "test_mc_lj_disp.h5"
filename_langevin = "test_mc_lj_langevin.h5"

import os

if os.path.isfile(filename_barostat):
    os.remove(filename_barostat)
reporter_barostat = MCReporter(filename_barostat, 1)

if os.path.isfile(filename_displacement):
    os.remove(filename_displacement)
reporter_displacement = MCReporter(filename_displacement, 10)

if os.path.isfile(filename_langevin):
    os.remove(filename_langevin)
reporter_langevin = LangevinDynamicsReporter(filename_langevin, 10)

from chiron.mcmc import (
    MetropolisDisplacementMove,
    MonteCarloBarostatMove,
    LangevinDynamicsMove,
    MoveSchedule,
    MCMCSampler,
)

mc_displacement_move = MetropolisDisplacementMove(
    displacement_sigma=0.001 * unit.nanometer,
    nr_of_moves=100,
    reporter=reporter_displacement,
    report_frequency=10,
    update_stepsize=True,
    update_stepsize_frequency=100,
)

mc_barostat_move = MonteCarloBarostatMove(
    volume_max_scale=0.1,
    nr_of_moves=10,
    reporter=reporter_barostat,
    report_frequency=1,
    update_stepsize=True,
    update_stepsize_frequency=50,
)

langevin_dynamics_move = LangevinDynamicsMove(
    stepsize=1.0 * unit.femtoseconds,
    collision_rate=1.0 / unit.picoseconds,
    nr_of_steps=100,
    reporter=reporter_langevin,
    report_frequency=10,
    reinitialize_velocities=True,
)


move_set = MoveSchedule(
    [
        ("LangevinDynamicsMove", langevin_dynamics_move),
        ("MetropolisDisplacementMove", mc_displacement_move),
        ("MonteCarloBarostatMove", mc_barostat_move),
    ]
)

sampler = MCMCSampler(move_set, sampler_state, thermodynamic_state)
sampler.run(n_iterations=10, nbr_list=nbr_list)  # how many times to repeat
