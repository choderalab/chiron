from openmmtools.testsystems import LennardJonesFluid

# Use the LennardJonesFluid example from openmmtools to initialize particle positions and topology
# For this example, the topology provides the masses for the particles
# The default LennardJonesFluid example considers the system to be Argon with 39.9 amu
lj_fluid = LennardJonesFluid(reduced_density=0.1, nparticles=1000)


from chiron.potential import LJPotential
from openmm import unit
from chiron.utils import PRNG, initialize_velocities


# initialize the LennardJones potential in chiron
#
sigma = 0.34 * unit.nanometer
epsilon = 0.238 * unit.kilocalories_per_mole
cutoff = 3.0 * sigma

lj_potential = LJPotential(
    lj_fluid.topology, sigma=sigma, epsilon=epsilon, cutoff=cutoff
)


from chiron.states import SamplerState, ThermodynamicState

PRNG.set_seed(1234)
# define the sampler state
sampler_state = SamplerState(
    x0=lj_fluid.positions,
    current_PRNG_key=PRNG.get_random_key(),
    box_vectors=lj_fluid.system.getDefaultPeriodicBoxVectors(),
)

velocities = initialize_velocities(
    300 * unit.kelvin, lj_fluid.topology, PRNG.get_random_key()
)

print(velocities)

# define the thermodynamic state
thermodynamic_state = ThermodynamicState(
    potential=lj_potential,
    temperature=300 * unit.kelvin,
)


from chiron.neighbors import NeighborListNsqrd, OrthogonalPeriodicSpace

# define the neighbor list for an orthogonal periodic space
skin = 0.5 * unit.nanometer

nbr_list = NeighborListNsqrd(
    OrthogonalPeriodicSpace(), cutoff=cutoff, skin=skin, n_max_neighbors=180
)

# build the neighbor list from the sampler state
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
integrator = LangevinIntegrator(reporter=reporter, report_frequency=100)
print("init_energy: ", lj_potential.compute_energy(sampler_state.x0, nbr_list))

updated_sampler_state = integrator.run(
    sampler_state,
    thermodynamic_state,
    n_steps=5000,
    nbr_list=nbr_list,
    progress_bar=True,
    initialize_velocities=True,
)

import h5py

# read the data from the reporter
with h5py.File("test_lj.h5", "r") as f:
    energies = f["potential_energy"][:]
    steps = f["step"][:]


# plot the energy
import matplotlib.pyplot as plt

plt.plot(steps, energies)
plt.xlabel("Step (fs)")
plt.ylabel("Energy (kj/mol)")
plt.show()

print(energies)
print(steps)
