from openmmtools.testsystems import LennardJonesFluid

# Use the LennardJonesFluid example from openmmtools to initialize particle positions and topology
# For this example, the topology provides the masses for the particles
# The default LennardJonesFluid example considers the system to be Argon with 39.9 amu
lj_fluid = LennardJonesFluid(reduced_density=0.005, nparticles=1000)


from chiron.potential import LJPotential
from openmm import unit

# initialize the LennardJones potential in chiron
#
sigma = 0.34 * unit.nanometer
epsilon = 0.238 * unit.kilocalories_per_mole
cutoff = 3.0 * sigma

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
    temperature=300 * unit.kelvin,
    pressure=0.5 * unit.atmosphere,
)

from chiron.neighbors import NeighborListNsqrd, OrthogonalPeriodicSpace

# define the neighbor list for an orthogonal periodic space
skin = 0.5 * unit.nanometer

nbr_list = NeighborListNsqrd(
    OrthogonalPeriodicSpace(), cutoff=cutoff, skin=skin, n_max_neighbors=180
)


# build the neighbor list from the sampler state
nbr_list.build_from_state(sampler_state)

from chiron.reporters import SimulationReporter

# initialize a reporter to save the simulation data
filename1 = "test_lj_mc.h5"
filename2 = "test_lj_langevin.h5"
import os

if os.path.isfile(filename1):
    os.remove(filename1)
if os.path.isfile(filename2):
    os.remove(filename2)
reporter1 = SimulationReporter(filename1, lj_fluid.topology, 10)
reporter2 = SimulationReporter(filename2, lj_fluid.topology, 10)

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
    simulation_reporter=reporter2,
    seed=1234,
)

mc_disp_move = MetropolisDisplacementMove(
    seed=1234,
    displacement_sigma=0.01 * unit.nanometer,
    nr_of_moves=1000,
    simulation_reporter=reporter1,
)

mc_barostat_move = MCBarostatMove(
    seed=1234,
    volume_max_scale=0.01,
    nr_of_moves=100,
    simulation_reporter=reporter1,
)
move_set = MoveSet(
    [
        ("MetropolisDisplacementMove", mc_disp_move),
        ("LangevinMove", langevin_move),
        ("MCBarostatMove", mc_barostat_move),
    ]
)

sampler = MCMCSampler(move_set, sampler_state, thermodynamic_state)

mass = unit.Quantity(39.948, unit.gram / unit.mole)
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

sampler.run(n_iterations=100, nbr_list=nbr_list)  # how many times to repeat

final_density = (
    mass
    * sampler.sampler_state.x0.shape[0]
    / unit.AVOGADRO_CONSTANT_NA
    / (sampler.thermodynamic_state.volume)
)
print(f"initial density: {initial_density}\nfinal density: {final_density}")
print(
    f"initial density: {initial_density.value_in_unit(unit.kilogram/unit.meter**3)}\nfinal density: {final_density.value_in_unit(unit.kilogram/unit.meter**3)}"
)
print(mc_barostat_move.statistics)
print(mc_disp_move.statistics)
