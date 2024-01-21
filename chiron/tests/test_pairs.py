import jax.numpy as jnp
import pytest
from chiron.neighbors import (
    NeighborListNsqrd,
    PairList,
    OrthogonalPeriodicSpace,
    OrthogonalNonperiodicSpace,
)
from chiron.states import SamplerState

from openmm import unit


def test_orthogonal_periodic_displacement():
    # test that the incorrect box shapes throw an exception
    with pytest.raises(ValueError):
        space = OrthogonalPeriodicSpace(jnp.array([10.0, 10.0, 10.0]))
    # test that incorrect units throw an exception
    with pytest.raises(ValueError):
        space = OrthogonalPeriodicSpace(
            unit.Quantity(
                jnp.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]),
                unit.radians,
            )
        )

    space = OrthogonalPeriodicSpace(
        jnp.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    )

    # test that the box vectors are set correctly
    assert jnp.all(
        space.box_vectors
        == jnp.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    )

    # test that the box lengths for an orthogonal box are set correctly
    assert jnp.all(space._box_lengths == jnp.array([10.0, 10.0, 10.0]))

    # test calculation of the displacement_vector and distance between two points
    p1 = jnp.array([[0, 0, 0], [0, 0, 0]])
    p2 = jnp.array([[1, 0, 0], [6, 0, 0]])

    r_ij, distance = space.displacement(p1, p2)

    assert jnp.all(r_ij == jnp.array([[-1.0, 0.0, 0.0], [4.0, 0.0, 0.0]]))

    assert jnp.all(distance == jnp.array([1, 4]))

    # test that the periodic wrapping works as expected
    wrapped_x = space.wrap(jnp.array([11, 0, 0]))
    assert jnp.all(wrapped_x == jnp.array([1, 0, 0]))

    wrapped_x = space.wrap(jnp.array([-1, 0, 0]))
    assert jnp.all(wrapped_x == jnp.array([9, 0, 0]))

    wrapped_x = space.wrap(jnp.array([5, 0, 0]))
    assert jnp.all(wrapped_x == jnp.array([5, 0, 0]))

    wrapped_x = space.wrap(jnp.array([5, 12, -1]))
    assert jnp.all(wrapped_x == jnp.array([5, 2, 9]))

    # test the setter for the box vectors
    space.box_vectors = jnp.array(
        [[10.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 30.0]]
    )
    assert jnp.all(
        space._box_vectors
        == jnp.array([[10.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 30.0]])
    )
    assert jnp.all(space._box_lengths == jnp.array([10.0, 20.0, 30.0]))


def test_orthogonal_nonperiodic_displacement():
    space = OrthogonalNonperiodicSpace(
        jnp.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    )
    p1 = jnp.array([[0, 0, 0], [0, 0, 0]])
    p2 = jnp.array([[1, 0, 0], [6, 0, 0]])

    r_ij, distance = space.displacement(p1, p2)

    assert jnp.all(r_ij == jnp.array([[-1.0, 0.0, 0.0], [-6.0, 0.0, 0.0]]))

    assert jnp.all(distance == jnp.array([1, 6]))

    wrapped_x = space.wrap(jnp.array([11, -1, 2]))
    assert jnp.all(wrapped_x == jnp.array([11, -1, 2]))


def test_neighborlist_pair():
    """
    This simple test of the neighborlist for 2 particles
    """

    coordinates = jnp.array([[0, 0, 0], [1, 0, 0]])
    box_vectors = jnp.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    from chiron.utils import PRNG

    PRNG.set_seed(1234)

    state = SamplerState(
        x0=unit.Quantity(coordinates, unit.nanometer),
        current_PRNG_key=PRNG.get_random_key(),
        box_vectors=unit.Quantity(box_vectors, unit.nanometer),
    )

    space = OrthogonalPeriodicSpace()
    cutoff = 1.1
    skin = 0.1
    nbr_list = NeighborListNsqrd(
        space,
        cutoff=unit.Quantity(cutoff, unit.nanometer),
        skin=unit.Quantity(skin, unit.nanometer),
        n_max_neighbors=5,
    )
    assert nbr_list.cutoff == cutoff
    assert nbr_list.skin == skin
    assert nbr_list.cutoff_and_skin == cutoff + skin
    assert nbr_list.n_max_neighbors == 5

    nbr_list.build_from_state(state)

    assert jnp.all(nbr_list.ref_coordinates == coordinates)
    assert jnp.all(nbr_list.box_vectors == box_vectors)
    assert nbr_list.is_built == True

    nbr_list.build(state.x0, state.box_vectors)

    assert jnp.all(nbr_list.ref_coordinates == coordinates)
    assert jnp.all(nbr_list.box_vectors == box_vectors)
    assert nbr_list.is_built == True

    # padded array of length 5
    assert jnp.all(
        nbr_list.neighbor_list == jnp.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])
    )

    # we won't double count pairs, so only particle 1 will have a neighbor
    assert jnp.all(nbr_list.n_neighbors == jnp.array([1, 0]))

    assert jnp.all(
        nbr_list.neighbor_mask == jnp.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    )

    n_neighbors, neighbor_list, padding_mask, dist, r_ij = nbr_list.calculate(
        coordinates
    )
    assert jnp.all(n_neighbors == jnp.array([1, 0]))

    # 2 particles, padded to 5
    assert jnp.all(neighbor_list.shape == (2, 5))

    assert jnp.all(neighbor_list == jnp.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]))

    assert jnp.all(padding_mask == jnp.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))

    assert jnp.all(
        dist == jnp.array([[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
    )

    assert jnp.all(
        r_ij
        == jnp.array(
            [
                [
                    [-1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ],
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
            ]
        )
    )

    # we haven't moved anything so we shouldn't need to rebuild
    assert nbr_list.check(coordinates) == False
    # shift coordinates, which should require a rebuild
    coordinates = coordinates + 0.1
    assert nbr_list.check(coordinates) == True

    coordinates = jnp.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    # we changed number of particles, and thus should rebuild
    assert nbr_list.check(coordinates) == True


def test_inputs():
    space = OrthogonalPeriodicSpace()
    # every particle should interact with every other particle
    cutoff = 2.1
    skin = 0.1
    nbr_list = NeighborListNsqrd(
        space,
        cutoff=unit.Quantity(cutoff, unit.nanometer),
        skin=unit.Quantity(skin, unit.nanometer),
        n_max_neighbors=5,
    )
    # check that the state is of the correct type
    with pytest.raises(TypeError):
        nbr_list.build_from_state(123)

    coordinates = jnp.array([[1, 2, 3], [0, 0, 0]])
    from chiron.utils import PRNG

    PRNG.set_seed(1234)

    state = SamplerState(
        x0=unit.Quantity(coordinates, unit.nanometer),
        current_PRNG_key=PRNG.get_random_key(),
        box_vectors=None,
    )

    # check that boxvectors are defined in the state
    with pytest.raises(ValueError):
        nbr_list.build_from_state(state)

    box_vectors = jnp.array(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0], [0.0, 0.0, 10.0]]
    )

    # test the shape of the box vectors
    with pytest.raises(ValueError):
        nbr_list.build(coordinates, box_vectors)

    box_vectors = jnp.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

    # test units of coordinates
    with pytest.raises(ValueError):
        nbr_list.build(unit.Quantity(coordinates, unit.radians), box_vectors)

    # test units of box vector
    with pytest.raises(ValueError):
        nbr_list.build(
            unit.Quantity(coordinates, unit.nanometers),
            unit.Quantity(box_vectors, unit.radians),
        )

    # check type of space
    with pytest.raises(TypeError):
        NeighborListNsqrd(
            123,
            cutoff=unit.Quantity(cutoff, unit.nanometer),
            skin=unit.Quantity(skin, unit.nanometer),
            n_max_neighbors=5,
        )
    # check units of cutoff
    with pytest.raises(ValueError):
        NeighborListNsqrd(
            space,
            cutoff=unit.Quantity(cutoff, unit.radian),
            skin=unit.Quantity(skin, unit.nanometer),
            n_max_neighbors=5,
        )
    # check units of skin
    with pytest.raises(ValueError):
        NeighborListNsqrd(
            space,
            cutoff=unit.Quantity(cutoff, unit.nanometer),
            skin=unit.Quantity(skin, unit.radian),
            n_max_neighbors=5,
        )


def test_neighborlist_pair_multiple_particles():
    """
    Test the neighborlist for multiple particles
    """
    n_xyz = 2
    scale_factor = 2.0

    coord_mesh = jnp.mgrid[0:n_xyz, 0:n_xyz, 0:n_xyz] * scale_factor / n_xyz

    # transform the mesh into a list of coordinates shape (n_atoms, 3)
    coordinates = jnp.stack(coord_mesh.reshape(3, -1), axis=1, dtype=jnp.float32)

    box_vectors = jnp.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    from chiron.utils import PRNG

    PRNG.set_seed(1234)

    state = SamplerState(
        x0=unit.Quantity(coordinates, unit.nanometer),
        current_PRNG_key=PRNG.get_random_key(),
        box_vectors=unit.Quantity(box_vectors, unit.nanometer),
    )

    space = OrthogonalPeriodicSpace()
    # every particle should interact with every other particle
    cutoff = 2.1
    skin = 0.1
    nbr_list = NeighborListNsqrd(
        space,
        cutoff=unit.Quantity(cutoff, unit.nanometer),
        skin=unit.Quantity(skin, unit.nanometer),
        n_max_neighbors=5,
    )
    nbr_list.build_from_state(state)

    assert jnp.all(nbr_list.n_neighbors == jnp.array([7, 6, 5, 4, 3, 2, 1, 0]))

    n_interacting, neighbor_list, mask, dist, rij = nbr_list.calculate(coordinates)
    assert jnp.all(n_interacting == jnp.array([7, 6, 5, 4, 3, 2, 1, 0]))

    # every particle should be in the nieghbor list, but only a subset in the interacting range
    cutoff = 1.1
    skin = 1.1
    nbr_list = NeighborListNsqrd(
        space,
        cutoff=unit.Quantity(cutoff, unit.nanometer),
        skin=unit.Quantity(skin, unit.nanometer),
        n_max_neighbors=5,
    )
    nbr_list.build_from_state(state)

    assert jnp.all(nbr_list.n_neighbors == jnp.array([7, 6, 5, 4, 3, 2, 1, 0]))

    n_interacting, neighbor_list, mask, dist, rij = nbr_list.calculate(coordinates)
    assert jnp.all(n_interacting == jnp.array([3, 2, 2, 1, 2, 1, 1, 0]))
    assert jnp.all(neighbor_list.shape == (8, 17))

    assert jnp.all(
        neighbor_list
        == jnp.array(
            [
                [1, 2, 3, 4, 5, 6, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [2, 3, 4, 5, 6, 7, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                [3, 4, 5, 6, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                [4, 5, 6, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                [5, 6, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                [6, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
    )
    # test passing coordinates and box vectors directly
    nbr_list.build(state.x0, state.box_vectors)

    assert jnp.all(nbr_list.n_neighbors == jnp.array([7, 6, 5, 4, 3, 2, 1, 0]))

    n_interacting, neighbor_list, mask, dist, rij = nbr_list.calculate(coordinates)
    assert jnp.all(n_interacting == jnp.array([3, 2, 2, 1, 2, 1, 1, 0]))


def test_pairlist_pair():
    """
    This simple test of the neighborlist for 2 particles
    """

    coordinates = jnp.array([[0, 0, 0], [1, 0, 0]])
    box_vectors = jnp.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    from chiron.utils import PRNG

    PRNG.set_seed(1234)

    state = SamplerState(
        x0=unit.Quantity(coordinates, unit.nanometer),
        current_PRNG_key=PRNG.get_random_key(),
        box_vectors=unit.Quantity(box_vectors, unit.nanometer),
    )

    space = OrthogonalPeriodicSpace()
    cutoff = 1.1
    skin = 0.1
    pair_list = PairList(
        space,
        cutoff=unit.Quantity(cutoff, unit.nanometer),
    )

    assert pair_list.cutoff == cutoff

    pair_list.build_from_state(state)
    assert jnp.all(pair_list.all_pairs == jnp.array([[1], [0]], dtype=jnp.int32))
    assert jnp.all(pair_list.reduction_mask == jnp.array([[True], [False]]))
    assert pair_list.is_built == True

    n_pairs, all_pairs, mask, dist, displacement = pair_list.calculate(coordinates)

    assert jnp.all(n_pairs == jnp.array([1, 0]))
    assert jnp.all(all_pairs.shape == (2, 1))
    assert jnp.all(all_pairs == jnp.array([[1], [0]]))
    assert jnp.all(mask == jnp.array([[1], [0]]))
    assert jnp.all(dist == jnp.array([[1.0], [1.0]]))
    assert displacement.shape == (2, 1, 3)
    assert jnp.all(displacement == jnp.array([[[-1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]))

    assert pair_list.check(coordinates) == False

    coordinates = coordinates = jnp.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    # we changed number of particles, and thus should rebuild
    assert pair_list.check(coordinates) == True


def test_pair_list_multiple_particles():
    # test the pair list for multiple particles
    # will compare to neighborlist
    n_xyz = 2
    scale_factor = 2.0

    coord_mesh = jnp.mgrid[0:n_xyz, 0:n_xyz, 0:n_xyz] * scale_factor / n_xyz

    # transform the mesh into a list of coordinates shape (n_atoms, 3)
    coordinates = jnp.stack(coord_mesh.reshape(3, -1), axis=1, dtype=jnp.float32)

    box_vectors = jnp.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    from chiron.utils import PRNG

    PRNG.set_seed(1234)

    state = SamplerState(
        x0=unit.Quantity(coordinates, unit.nanometer),
        current_PRNG_key=PRNG.get_random_key(),
        box_vectors=unit.Quantity(box_vectors, unit.nanometer),
    )

    space = OrthogonalPeriodicSpace()
    # every particle should interact with every other particle
    cutoff = 2.1
    skin = 0.1
    pair_list = PairList(
        space,
        cutoff=unit.Quantity(cutoff, unit.nanometer),
    )
    pair_list.build_from_state(state)

    n_interacting, all_pairs, mask, dist, rij = pair_list.calculate(coordinates)
    assert jnp.all(n_interacting == jnp.array([7, 6, 5, 4, 3, 2, 1, 0]))
    assert jnp.all(all_pairs.shape == (8, 7))
    assert jnp.all(
        all_pairs
        == jnp.array(
            [
                [1, 2, 3, 4, 5, 6, 7],
                [0, 2, 3, 4, 5, 6, 7],
                [0, 1, 3, 4, 5, 6, 7],
                [0, 1, 2, 4, 5, 6, 7],
                [0, 1, 2, 3, 5, 6, 7],
                [0, 1, 2, 3, 4, 6, 7],
                [0, 1, 2, 3, 4, 5, 7],
                [0, 1, 2, 3, 4, 5, 6],
            ]
        )
    )
    assert jnp.all(mask.shape == (coordinates.shape[0], coordinates.shape[0] - 1))

    # compare to nbr_list
    nbr_list = NeighborListNsqrd(
        space,
        cutoff=unit.Quantity(cutoff, unit.nanometer),
        skin=unit.Quantity(skin, unit.nanometer),
        n_max_neighbors=20,
    )
    nbr_list.build_from_state(state)
    n_interacting1, all_pairs, mask1, dist1, rij1 = nbr_list.calculate(coordinates)

    # sum up all the distances within range, see if they match those in the nlist
    assert jnp.where(mask, dist, 0).sum() == jnp.where(mask1, dist1, 0).sum()

    assert jnp.where(
        dist
        == jnp.array(
            [
                [1.0, 1.0, 1.4142135, 1.0, 1.4142135, 1.4142135, 1.7320508],
                [1.0, 1.4142135, 1.0, 1.4142135, 1.0, 1.7320508, 1.4142135],
                [1.0, 1.4142135, 1.0, 1.4142135, 1.7320508, 1.0, 1.4142135],
                [1.4142135, 1.0, 1.0, 1.7320508, 1.4142135, 1.4142135, 1.0],
                [1.0, 1.4142135, 1.4142135, 1.7320508, 1.0, 1.0, 1.4142135],
                [1.4142135, 1.0, 1.7320508, 1.4142135, 1.0, 1.4142135, 1.0],
                [1.4142135, 1.7320508, 1.0, 1.4142135, 1.0, 1.4142135, 1.0],
                [1.7320508, 1.4142135, 1.4142135, 1.0, 1.4142135, 1.0, 1.0],
            ],
        )
    )
