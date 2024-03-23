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


from chiron.utils import PRNG

PRNG.set_seed(1234)

from chiron.states import SamplerState, ThermodynamicState

# define the sampler state
sampler_state = SamplerState(
    positions=lj_fluid.positions,
    current_PRNG_key=PRNG.get_random_key(),
    box_vectors=lj_fluid.system.getDefaultPeriodicBoxVectors(),
)

# define the thermodynamic state
thermodynamic_state = ThermodynamicState(
    potential=lj_potential,
    temperature=300 * unit.kelvin,
)


from chiron.neighbors import NeighborListNsqrd, OrthogonalPeriodicSpace

# Set up a neighbor list for an orthogonal periodic box with a cutoff of 3.0 * sigma and skin of 0.5 * sigma,
# where sigma = 0.34 nm.
# The class we instantiate, NeighborListNsqrd, uses an O(N^2) calculation to build the neighbor list,
# but uses a buffer (i.e., the skin) to avoid needing to perform the O(N^2) calculation at every step.
# With this routine, the calculation at each step between builds is O(N*n_max_neighbors).
# For the conditions considered here, n_max_neighbors is set to 180 (note this will increase if necessary)
# and thus there is ~5 reduction in computational cost compared to a brute force approach (i.e., PairListNsqrd).

skin = 0.5 * unit.nanometer

nbr_list = NeighborListNsqrd(
    OrthogonalPeriodicSpace(), cutoff=cutoff, skin=skin, n_max_neighbors=180
)

# perform the initial build of the neighbor list from the sampler state
nbr_list.build_from_state(sampler_state)

from chiron.reporters import LangevinDynamicsReporter

# initialize a reporter to save the simulation data
filename = "test_lj.h5"
import os

if os.path.isfile(filename):
    os.remove(filename)
reporter = LangevinDynamicsReporter(
    "test_lj.h5",
    1,
    lj_fluid.topology,
)

from chiron.integrators import LangevinIntegrator

# initialize the Langevin integrator
integrator = LangevinIntegrator(reporter=reporter, report_interval=100)
print("init_energy: ", lj_potential.compute_energy(sampler_state.positions, nbr_list))

# run the simulation
# note, typically we will not be calling the integrator directly,
# but instead using the LangevinDynamics Move in the MCMC Sampler.
updated_sampler_state, updated_nbr_list = integrator.run(
    sampler_state,
    thermodynamic_state,
    number_of_steps=1000,
    nbr_list=nbr_list,
    progress_bar=True,
)

import h5py

# read the data from the reporter
with h5py.File("test_lj.h5", "r") as f:
    energies = f["potential_energy"][:]
    steps = f["step"][:]

energies = reporter.get_property("potential_energy")
steps = reporter.get_property("step")

# plot the energy
import matplotlib.pyplot as plt

plt.plot(steps, energies)
plt.xlabel("Step (fs)")
plt.ylabel("Energy (kj/mol)")
plt.show()

print(energies)
print(steps)
