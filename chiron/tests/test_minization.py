def test_minimization():
    from chiron.minimze import minimize_energy
    import jax.numpy as jnp

    from chiron.states import SamplerState
    from chiron.neighbors import PairList, OrthogonalPeriodicSpace
    from openmm import unit

    # initialize testystem
    from openmmtools.testsystems import LennardJonesFluid

    lj_fluid = LennardJonesFluid(reduced_density=0.1, n_particles=200)
    # initialize potential
    from chiron.potential import LJPotential

    cutoff = unit.Quantity(1.0, unit.nanometer)
    lj_potential = LJPotential(lj_fluid.topology, cutoff=cutoff)

    sampler_state = SamplerState(
        lj_fluid.positions, box_vectors=lj_fluid.system.getDefaultPeriodicBoxVectors()
    )
    # use parilist
    nbr_list = PairList(OrthogonalPeriodicSpace(), cutoff=cutoff)
    nbr_list.build_from_state(sampler_state)

    # compute intial energy with and without pairlist
    initial_e_with_nbr_list = lj_potential.compute_energy(sampler_state.x0, nbr_list)
    initial_e_without_nbr_list = lj_potential.compute_energy(sampler_state.x0)
    print(f"initial_e_with_nbr_list: {initial_e_with_nbr_list}")
    print(f"initial_e_without_nbr_list: {initial_e_without_nbr_list}")
    assert not jnp.isclose(
        initial_e_with_nbr_list, initial_e_without_nbr_list
    ), "initial_e_with_nbr_list and initial_e_without_nbr_list should not be close"
    # minimize energy for 0 steps
    results = minimize_energy(
        sampler_state.x0, lj_potential.compute_energy, nbr_list, maxiter=0
    )

    # check that the minimization did not change the energy
    min_x = results.params
    # after 0 steps of minimization
    after_0_steps_minimization_e_with_nbr_list = lj_potential.compute_energy(
        min_x, nbr_list
    )
    after_0_steps_minimization_e_without_nbr_list = lj_potential.compute_energy(
        sampler_state.x0
    )
    print(
        f"after_0_steps_minimization_e_with_nbr_list: {after_0_steps_minimization_e_with_nbr_list}"
    )
    print(
        f"after_0_steps_minimization_e_without_nbr_list: {after_0_steps_minimization_e_without_nbr_list}"
    )
    assert jnp.isclose(
        initial_e_with_nbr_list, after_0_steps_minimization_e_with_nbr_list
    )

    assert jnp.isclose(
        initial_e_without_nbr_list, after_0_steps_minimization_e_without_nbr_list
    )

    # after 100 steps of minimization
    steps = 100
    results = minimize_energy(
        sampler_state.x0, lj_potential.compute_energy, nbr_list, maxiter=steps
    )
    min_x = results.params
    e_min = lj_potential.compute_energy(min_x, nbr_list)
    print(f"e_min after {steps} of minimization: {e_min}")
    # test that e_min is smaller than initial_e_with_nbr_list
    assert e_min < initial_e_with_nbr_list
    # test that e is not Nan
    assert not jnp.isnan(lj_potential.compute_energy(min_x, nbr_list))