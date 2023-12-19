def test_minimization():
    from chiron.minimze import minimize_energy
    import jax.numpy as jnp

    from chiron.states import SamplerState
    from chiron.neighbors import NeighborListNsqrd, OrthogonalPeriodicSpace
    from openmm import unit

    # initialize testystem
    from openmmtools.testsystems import LennardJonesFluid

    lj_fluid = LennardJonesFluid(reduced_density=0.1, n_particles=100)
    # initialize potential
    from chiron.potential import LJPotential

    cutoff = unit.Quantity(1., unit.nanometer)
    lj_potential = LJPotential(lj_fluid.topology, cutoff=cutoff)

    sampler_state = SamplerState(
        lj_fluid.positions, box_vectors=lj_fluid.system.getDefaultPeriodicBoxVectors()
    )
    skin = unit.Quantity(0.1, unit.nanometer)

    nbr_list = NeighborListNsqrd(
        OrthogonalPeriodicSpace(), cutoff=cutoff, skin=skin, n_max_neighbors=180
    )
    nbr_list.build_from_state(sampler_state)

    print(lj_potential.compute_energy(sampler_state.x0, nbr_list))
    print(lj_potential.compute_energy(sampler_state.x0))

    min_x = minimize_energy(sampler_state.x0, lj_potential.compute_energy, nbr_list)
    e = lj_potential.compute_energy(min_x, nbr_list)
    assert jnp.isclose(e, -12506.332)    