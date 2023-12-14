import jax.numpy as jnp
import pytest
from chiron.neighbors import NeighborListNsqrd, OrthogonalPeriodicSpace, OrthogonalNonperiodicSpace
from chiron.states import SamplerState

from openmm import unit

def test_orthogonal_periodic_displacement():
    space = OrthogonalPeriodicSpace(jnp.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]))

    assert jnp.all(space._box_lengths == jnp.array([10.0, 10.0, 10.0]))

    p1 = jnp.array([[0, 0, 0], [0,0,0]])
    p2 = jnp.array([[1, 0, 0], [6,0,0]])

    r_ij, distance = space.displacement(p1, p2)

    assert jnp.all(r_ij == jnp.array([[-1.,  0.,  0.],
        [ 4.,  0.,  0.]]))

    assert jnp.all(distance == jnp.array([1,4]) )

    wrapped_x = space.wrap(jnp.array([11, 0, 0]))
    assert jnp.all(wrapped_x == jnp.array([1,0,0]))

    wrapped_x = space.wrap(jnp.array([-1, 0, 0]))
    assert jnp.all(wrapped_x == jnp.array([9, 0, 0]))

    wrapped_x = space.wrap(jnp.array([5, 0, 0]))
    assert jnp.all(wrapped_x == jnp.array([5, 0, 0]))

    wrapped_x = space.wrap(jnp.array([5, 12, -1]))
    assert jnp.all(wrapped_x == jnp.array([5, 2, 9]))

    space.box_vectors = jnp.array([[10.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 30.0]])
    assert jnp.all(space._box_vectors == jnp.array([[10.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 30.0]]))
    assert jnp.all(space._box_lengths == jnp.array([10.0, 20.0, 30.0]))

def test_orthogonal_nonperiodic_displacement():
    space = OrthogonalNonperiodicSpace(jnp.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]))
    p1 = jnp.array([[0, 0, 0], [0, 0, 0]])
    p2 = jnp.array([[1, 0, 0], [6, 0, 0]])

    r_ij, distance = space.displacement(p1, p2)

    assert jnp.all(r_ij == jnp.array([[-1., 0., 0.],
                                      [-6., 0., 0.]]))

    assert jnp.all(distance == jnp.array([1, 6]))

    wrapped_x = space.wrap(jnp.array([11, -1, 2]) )
    assert jnp.all(wrapped_x == jnp.array([11,-1,2]))
def test_neighborlist_pair():
    """
    This simple test test aspects of the neighborlist for 2 particles
    """

    coordinates = jnp.array([[0, 0, 0], [1, 0, 0]])
    box_vectors = jnp.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    state = SamplerState(x0=unit.Quantity(coordinates, unit.nanometer),
                         box_vectors=unit.Quantity(box_vectors,
                                                   unit.nanometer))

    space = OrthogonalPeriodicSpace()
    cutoff = 1.1
    skin = 0.1
    nbr_list = NeighborListNsqrd(space, cutoff = unit.Quantity(cutoff, unit.nanometer), skin=unit.Quantity(skin, unit.nanometer), n_max_neighbors=5)
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
    assert jnp.all(nbr_list.neighbor_list == jnp.array([[1, 1, 1, 1, 1],
       [0, 0, 0, 0, 0]]))

    # we won't double count pairs, so only particle 1 will have a neighbor
    assert jnp.all(nbr_list.n_neighbors == jnp.array([1, 0]) )

    assert jnp.all(nbr_list.neighbor_mask == jnp.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))

    n_neighbors, padding_mask, dist, r_ij = nbr_list.calculate(coordinates)
    assert jnp.all(n_neighbors == jnp.array([1, 0]))
    assert jnp.all(padding_mask == jnp.array([[1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]]))

    assert jnp.all(dist == jnp.array([[1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.]]))

    assert jnp.all(r_ij == jnp.array([[[-1.,  0.,  0.],
         [-1.,  0.,  0.],
         [-1.,  0.,  0.],
         [-1.,  0.,  0.],
         [-1.,  0.,  0.]],

        [[ 1.,  0.,  0.],
         [ 1.,  0.,  0.],
         [ 1.,  0.,  0.],
         [ 1.,  0.,  0.],
         [ 1.,  0.,  0.]]]))

    # we haven't moved anything so we shouldn't need to rebuild
    assert nbr_list.check(coordinates) == False
    # shift coordinates, which should require a rebuild
    coordinates = coordinates + 0.1
    assert nbr_list.check(coordinates) == True
def test_neighborlist_pair2():
    n_xyz = 2
    scale_factor = 2.0

    coord_mesh = jnp.mgrid[0:n_xyz, 0:n_xyz, 0:n_xyz] * scale_factor / n_xyz

    # transform the mesh into a list of coordinates shape (n_atoms, 3)
    coordinates = jnp.stack(coord_mesh.reshape(3, -1), axis=1, dtype=jnp.float32)

    box_vectors = jnp.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    state = SamplerState(x0=unit.Quantity(coordinates, unit.nanometer),
                         box_vectors=unit.Quantity(box_vectors,
                                                   unit.nanometer))

    space = OrthogonalPeriodicSpace()
    # every particle should interact with every other particle
    cutoff = 2.1
    skin = 0.1
    nbr_list = NeighborListNsqrd(space, cutoff=unit.Quantity(cutoff, unit.nanometer),
                                 skin=unit.Quantity(skin, unit.nanometer), n_max_neighbors=5)
    nbr_list.build_from_state(state)

    assert jnp.all(nbr_list.n_neighbors == jnp.array([7, 6, 5, 4, 3, 2, 1, 0]))

    n_interacting, mask, dist, rij = nbr_list.calculate(coordinates)
    assert jnp.all(n_interacting == jnp.array([7, 6, 5, 4, 3, 2, 1, 0]))

    # every particle should be in the nieghbor list, but only a subset in the interacting range
    cutoff = 1.1
    skin = 1.1
    nbr_list = NeighborListNsqrd(space, cutoff=unit.Quantity(cutoff, unit.nanometer),
                                 skin=unit.Quantity(skin, unit.nanometer), n_max_neighbors=5)
    nbr_list.build_from_state(state)

    assert jnp.all(nbr_list.n_neighbors == jnp.array([7, 6, 5, 4, 3, 2, 1, 0]))

    n_interacting, mask, dist, rij = nbr_list.calculate(coordinates)
    assert jnp.all(n_interacting == jnp.array([3, 2, 2, 1, 2, 1, 1, 0]))

    nbr_list.build(state.x0, state.box_vectors)

    assert jnp.all(nbr_list.n_neighbors == jnp.array([7, 6, 5, 4, 3, 2, 1, 0]))

    n_interacting, mask, dist, rij = nbr_list.calculate(coordinates)
    assert jnp.all(n_interacting == jnp.array([3, 2, 2, 1, 2, 1, 1, 0]))