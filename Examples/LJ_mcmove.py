from openmmtools.testsystems import LennardJonesFluid

# Use the LennardJonesFluid example from openmmtools to initialize particle positions and topology
# For this example, the topology provides the masses for the particles
# The default LennardJonesFluid example considers the system to be Argon with 39.9 amu
lj_fluid = LennardJonesFluid(reduced_density=0.1, nparticles=1000)


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
from chiron.utils import PRNG

PRNG.set_seed(1234)

# define the sampler state
sampler_state = SamplerState(
    positions=lj_fluid.positions,
    current_PRNG_key=PRNG.get_random_key(),
    box_vectors=lj_fluid.system.getDefaultPeriodicBoxVectors(),
)

# define the thermodynamic state
thermodynamic_state = ThermodynamicState(
    potential=lj_potential, temperature=300 * unit.kelvin
)

from chiron.neighbors import NeighborListNsqrd, OrthogonalPeriodicSpace

# define the neighbor list for an orthogonal periodic space
skin = 0.5 * unit.nanometer

nbr_list = NeighborListNsqrd(
    OrthogonalPeriodicSpace(), cutoff=cutoff, skin=skin, n_max_neighbors=180
)
from chiron.neighbors import PairList


# build the neighbor list from the sampler state
nbr_list.build_from_state(sampler_state)

from chiron.reporters import MCReporter

# initialize a reporter to save the simulation data
filename = "test_mc_lj.h5"
import os

if os.path.isfile(filename):
    os.remove(filename)
reporter = MCReporter(filename, 1)

from chiron.mcmc import MonteCarloDisplacementMove

mc_move = MonteCarloDisplacementMove(
    displacement_sigma=0.01 * unit.nanometer,
    number_of_moves=5000,
    reporter=reporter,
    report_interval=1,
    autotune=True,
    autotune_interval=100,
)

mc_move.update(sampler_state, thermodynamic_state, nbr_list)

stats = mc_move.statistics
print(stats["n_accepted"] / stats["n_proposed"])


acceptance_probability = reporter.get_property("acceptance_probability")
displacement_sigma = reporter.get_property("displacement_sigma")
potential_energy = reporter.get_property("potential_energy")
step = reporter.get_property("step")

# plot the energy
import matplotlib.pyplot as plt

plt.subplot(3, 1, 1)

plt.plot(step, displacement_sigma)
plt.ylabel("displacement_sigma (nm)")

plt.subplot(3, 1, 2)

plt.plot(step, acceptance_probability)
plt.ylabel("acceptance_probability")


plt.subplot(3, 1, 3)

plt.plot(step, potential_energy)
plt.xlabel("Step")
plt.ylabel("potential_energy (kj/mol)")
plt.show()
