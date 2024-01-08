from openmmtools.testsystems import LennardJonesFluid

# Use the LennardJonesFluid example from openmmtools to initialize particle positions and topology
# For this example, the topology provides the masses for the particles
# The default LennardJonesFluid example considers the system to be Argon with 39.9 amu
lj_fluid = LennardJonesFluid(reduced_density=0.5, nparticles=1100)


from chiron.potential import LJPotential
from openmm import unit

# initialize the LennardJones potential for UA-TraPPE methane
#
sigma = 0.373 * unit.nanometer
epsilon = 0.2941 * unit.kilocalories_per_mole
cutoff = 1.4 * unit.nanometer

lj_potential = LJPotential(
    lj_fluid.topology, sigma=sigma, epsilon=epsilon, cutoff=cutoff
)

from chiron.states import SamplerState, ThermodynamicState

# define the sampler state
sampler_state = SamplerState(
    x0=lj_fluid.positions, box_vectors=lj_fluid.system.getDefaultPeriodicBoxVectors()
)

# define the thermodynamic state
thermodynamic_state = ThermodynamicState(
    potential=lj_potential,
    temperature=140 * unit.kelvin,
    pressure=13.00765 * unit.atmosphere,
)

from chiron.neighbors import PairList, OrthogonalPeriodicSpace

# define the neighbor list for an orthogonal periodic space
skin = 0.5 * unit.nanometer

nbr_list = PairList(OrthogonalPeriodicSpace(), cutoff=cutoff)


# build the neighbor list from the sampler state
nbr_list.build_from_state(sampler_state)

from chiron.minimze import minimize_energy

results = minimize_energy(
    sampler_state.x0, lj_potential.compute_energy, nbr_list, maxiter=100
)

min_x = results.params


from chiron.reporters import SimulationReporter

# initialize a reporter to save the simulation data
filename1 = "test_lj_nvt.h5"
filename2 = "test_lj_npt.h5"
filename3 = "test_lj_langevin.h5"
import os

if os.path.isfile(filename1):
    os.remove(filename1)
if os.path.isfile(filename2):
    os.remove(filename2)
if os.path.isfile(filename3):
    os.remove(filename3)
reporter1 = SimulationReporter(filename1, lj_fluid.topology, 10)
reporter2 = SimulationReporter(filename2, lj_fluid.topology, 10)
reporter3 = SimulationReporter(filename3, lj_fluid.topology, 10)

from chiron.mcmc import (
    MetropolisDisplacementMove,
    MoveSet,
    MCMCSampler,
    MCBarostatMove,
    LangevinDynamicsMove,
)

langevin_move = LangevinDynamicsMove(
    stepsize=1.0 * unit.femtoseconds,
    collision_rate=1.0 / unit.picoseconds,
    nr_of_steps=1000,
    simulation_reporter=reporter3,
    seed=1234,
)

mc_disp_move = MetropolisDisplacementMove(
    seed=1234,
    displacement_sigma=0.005 * unit.nanometer,
    nr_of_moves=90,
    simulation_reporter=reporter1,
)

mc_barostat_move = MCBarostatMove(
    seed=1234,
    volume_max_scale=0.01,
    nr_of_moves=10,
    simulation_reporter=reporter2,
)
move_set = MoveSet(
    [
        ("MetropolisDisplacementMove", mc_disp_move),
        # ("LangevinMove", langevin_move),
        ("MCBarostatMove", mc_barostat_move),
    ]
)

sampler = MCMCSampler(move_set, sampler_state, thermodynamic_state)

mass = unit.Quantity(16.04, unit.gram / unit.mole)
volume = (
    sampler_state.box_vectors[0][0]
    * sampler_state.box_vectors[1][1]
    * sampler_state.box_vectors[2][2]
)
initial_density = (
    mass
    * sampler.sampler_state.x0.shape[0]
    / unit.AVOGADRO_CONSTANT_NA
    / (unit.Quantity(volume, unit.nanometer**3))
)
densities = []
densities.append(initial_density)
sampler.run(n_iterations=20000, nbr_list=nbr_list)  # how many times to repeat

final_density = (
    mass
    * sampler.sampler_state.x0.shape[0]
    / unit.AVOGADRO_CONSTANT_NA
    / (sampler.thermodynamic_state.volume)
)
densities.append(final_density)

print(f"initial density: {initial_density}\nfinal density: {final_density}")
print(
    f"initial density: {initial_density.value_in_unit(unit.kilogram/unit.meter**3)}\nfinal density: {final_density.value_in_unit(unit.kilogram/unit.meter**3)}"
)
print(mc_barostat_move.statistics)
print(mc_disp_move.statistics)
