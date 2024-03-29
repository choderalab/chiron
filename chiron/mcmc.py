from chiron.states import SamplerState, ThermodynamicState
from openmm import unit
from typing import Tuple, List, Optional
import jax.numpy as jnp
from chiron.reporters import LangevinDynamicsReporter, _SimulationReporter, MCReporter
from .neighbors import PairsBase

from abc import ABC, abstractmethod


class MCMCMove:
    def __init__(
        self,
        number_of_moves: int,
        reporter: Optional[_SimulationReporter] = None,
        report_interval: Optional[int] = 100,
    ):
        """
        Initialize a move within the molecular system.

        Parameters
        ----------
        number_of_moves : int
            Number of moves to be applied.
        reporter : _SimulationReporter, optional
            Reporter object for saving the simulation data.
            Default is None.
        report_interval : int, optional
            Interval for saving the simulation data in the reporter.
            Default is 100.

        """

        self.number_of_moves = number_of_moves
        self.reporter = reporter
        self.report_interval = report_interval

        # we need to keep track of which iteration we are on
        self._move_iteration = 0

        # we also need to keep track of attempts made (i.e., total elapsed steps), in case the number_of_moves is changed
        self._number_of_attempts_made = 0

        from loguru import logger as log

        if self.reporter is not None:
            log.info(
                f"Using reporter {self.reporter} saving to {self.reporter.workdir}"
            )
            assert self.report_interval is not None

    @abstractmethod
    def update(
        self,
        sampler_state: SamplerState,
        thermodynamic_state: ThermodynamicState,
        nbr_list: Optional[PairsBase] = None,
    ) -> Tuple[SamplerState, ThermodynamicState, Optional[PairsBase]]:
        """
        Update the state of the system.

        Parameters
        ----------
        sampler_state : SamplerState
            The sampler state to run the integrator on.
        thermodynamic_state : ThermodynamicState
            The thermodynamic state to run the integrator on.
        nbr_list : PairsBase, optional
            The neighbor list to use for the simulation.

        Returns
        -------
        sampler_state : SamplerState
            The updated sampler state.
        thermodynamic_state : ThermodynamicState
            The updated thermodynamic state.
        nbr_list: PairsBase
            The updated neighbor/pair list. If no nbr_list is passed, this will be None.

        """
        pass

    @property
    def number_of_attemps_made(self):
        """
        Return the total number of steps that have been attempted in the  move.
        """
        return self._number_of_attempts_made


class LangevinDynamicsMove(MCMCMove):
    def __init__(
        self,
        timestep: unit.Quantity = 1.0 * unit.femtoseconds,
        collision_rate: unit.Quantity = 1.0 / unit.picoseconds,
        refresh_velocities: bool = False,
        reporter: Optional[LangevinDynamicsReporter] = None,
        report_interval: int = 100,
        number_of_steps: int = 1_000,
        save_traj_in_memory: bool = False,
    ):
        """
        Initialize the LangevinDynamicsMove with a molecular system.

        Parameters
        ----------
        timestep : unit.Quantity
            Time step size for the integration.
        collision_rate : unit.Quantity
            Collision rate for the Langevin dynamics.
        refresh_velocities : bool, optional
            Whether to reinitialize the velocities each time the run function is called.
            Default is False.
        reporter : LangevinDynamicsReporter, optional
            Reporter object for saving the simulation data.
            Default is None.
        report_interval : int
            Interval for saving the simulation data.
            Default is 100.
        number_of_steps : int, optional
            Number of steps to run the integrator for.
            Default is 1_000.
        save_traj_in_memory: bool
            Flag indicating whether to save the trajectory in memory.
            Default is False. NOTE: Only for debugging purposes.
        """
        super().__init__(
            number_of_moves=number_of_steps,
            reporter=reporter,
            report_interval=report_interval,
        )

        self.timestep = timestep
        self.collision_rate = collision_rate
        self.save_traj_in_memory = save_traj_in_memory
        self.traj = []
        from chiron.integrators import LangevinIntegrator

        self.integrator = LangevinIntegrator(
            timestep=self.timestep,
            collision_rate=self.collision_rate,
            refresh_velocities=refresh_velocities,
            report_interval=report_interval,
            reporter=reporter,
            save_traj_in_memory=save_traj_in_memory,
        )

    def update(
        self,
        sampler_state: SamplerState,
        thermodynamic_state: ThermodynamicState,
        nbr_list: Optional[PairsBase] = None,
    ) -> Tuple[SamplerState, ThermodynamicState, Optional[PairsBase]]:
        """
        Run the integrator to perform molecular dynamics simulation.

        Parameters
        ----------
        sampler_state : SamplerState
            The sampler state to run the integrator on.
        thermodynamic_state : ThermodynamicState
            The thermodynamic state to run the integrator on.
        nbr_list : PairsBase, optional
            The neighbor list to use for the simulation.

        Returns
        -------
        sampler_state : SamplerState
            The updated sampler state.
        thermodynamic_state : ThermodynamicState
            The thermodynamic state; note this is not modified by the Langevin dynamics algorithm.
        nbr_list: PairsBase
            The updated neighbor/pair list. If a nbr_list is not set, this will be None.
        """

        assert isinstance(
            sampler_state, SamplerState
        ), f"Sampler state must be SamplerState, not {type(sampler_state)}"
        assert isinstance(
            thermodynamic_state, ThermodynamicState
        ), f"Thermodynamic state must be ThermodynamicState, not {type(thermodynamic_state)}"

        updated_sampler_state, updated_nbr_list = self.integrator.run(
            thermodynamic_state=thermodynamic_state,
            sampler_state=sampler_state,
            number_of_steps=self.number_of_moves,
            nbr_list=nbr_list,
        )
        # update the elapsed steps
        self._number_of_attempts_made += self.number_of_moves

        if self.save_traj_in_memory:
            self.traj.append(self.integrator.traj)
            self.integrator.traj = []

        self._move_iteration += 1

        # The thermodynamic_state will not change for the langevin move
        return updated_sampler_state, thermodynamic_state, updated_nbr_list


class MCMove(MCMCMove):
    def __init__(
        self,
        number_of_moves: int,
        reporter: Optional[_SimulationReporter],
        report_interval: int = 1,
        autotune: bool = False,
        autotune_interval: int = 100,
        acceptance_method: str = "Metropolis-Hastings",
    ) -> None:
        """
        Initialize the move.

        Parameters
        ----------
        number_of_moves
            Number of moves to be attempted in each call to update.
        reporter
            Reporter object for saving the simulation step data.
        report_interval
            Interval for saving the simulation data.
        autotune
            Whether to automatically tune the parameters of the MC move to achieve a target acceptance ratio.
            For example, for a simple displacement move this would update the displacement_sigma.
        autotune_interval
            Frequency of autotuning the MC move parameters to achieve a target acceptance ratio.
        acceptance_method
            Methodology to use for accepting or rejecting the proposed state.
            Default is "Metropolis-Hastings".
        """
        super().__init__(
            number_of_moves=number_of_moves,
            reporter=reporter,
            report_interval=report_interval,
        )
        self.acceptance_method = acceptance_method  # I think we should pass a class/function instead of a string, like space.

        self.reset_statistics()
        self.autotune = autotune
        self.autotune_interval = autotune_interval

    def update(
        self,
        sampler_state: SamplerState,
        thermodynamic_state: ThermodynamicState,
        nbr_list: Optional[PairsBase] = None,
    ) -> Tuple[SamplerState, ThermodynamicState, Optional[PairsBase]]:
        """
        Perform the defined move and update the state.

        Parameters
        ----------
        sampler_state : SamplerState
            The initial state of the simulation, including positions.
        thermodynamic_state : ThermodynamicState
            The thermodynamic state of the system, including temperature and potential.
        nbr_list : PairBase, optional
            Neighbor list for the system.

        Returns
        -------
        sampler_state : SamplerState
            The updated sampler state.
        thermodynamic_state : ThermodynamicState
            The updated thermodynamic state.
        nbr_list: PairsBase
            The updated neighbor/pair list. If a nbr_list is not set, this will be None.
        """

        self._current_reduced_potential = None
        for i in range(self.number_of_moves):
            sampler_state, thermodynamic_state, nbr_list = self._step(
                sampler_state,
                thermodynamic_state,
                nbr_list,
            )
            self._number_of_attempts_made += 1

            # We should use self._number_of_attempts_made  as the  "step" otherwise, if we just used i, instances where
            # self.report_interval > self.number_of_moves would only report on the
            # first step, which might actually be more frequent than we specify

            if hasattr(self, "reporter"):
                if self.reporter is not None:
                    if self._number_of_attempts_made % self.report_interval == 0:
                        self._report(
                            i,
                            self._move_iteration,
                            self._number_of_attempts_made,
                            self.n_accepted / self.n_proposed,
                            sampler_state,
                            thermodynamic_state,
                            nbr_list,
                        )
            if self.autotune:
                # if we only used i, we might never actually update the parameters if we have a move that is called infrequently
                if (
                    self._number_of_attempts_made % self.autotune_interval == 0
                    and self._number_of_attempts_made > 0
                ):
                    self._autotune()
        # keep track of how many times this function has been called
        self._move_iteration += 1

        return sampler_state, thermodynamic_state, nbr_list

    @abstractmethod
    def _report(
        self,
        step: int,
        iteration: int,
        number_of_attempts_made: int,
        acceptance_probability: float,
        sampler_state: SamplerState,
        thermodynamic_state: ThermodynamicState,
        nbr_list: Optional[PairsBase] = None,
    ):
        """
        Report the current state of the MC move.

        Since different moves will be modifying different quantities,
        this needs to be defined for each move.

        Parameters
        ----------
        step : int
            The current step of the simulation move.
        iteration : int
            The current iteration of the move sequence (i.e., how many times has this been called thus far).
        number_of_attempts_made : int
            The total number of steps that have been taken in the simulation move. step+ nr_moves*iteration
        acceptance_probability : float
            The acceptance probability of the move.
        sampler_state : SamplerState
            The sampler state of the system.
        thermodynamic_state : ThermodynamicState
            The thermodynamic state of the system.
        nbr_list : Optional[PairBase]=None
            The neighbor list or pair list for evaluating interactions in the system, default None
        """
        pass

    @abstractmethod
    def _autotune(self):
        """
        This will autotune the move parameters to reach a target acceptance probability.
        This will be specific to the type of move, e.g., a displacement_sigma for a displacement move
        or a maximum volume change factor for a Monte Carlo barostat move.

        Since different moves will be modifying different quantities, this needs to be defined for each move.

        Note this will modify the class parameters in place.
        """
        pass

    def _step(
        self,
        current_sampler_state: SamplerState,
        current_thermodynamic_state: ThermodynamicState,
        current_nbr_list: Optional[PairsBase] = None,
    ) -> Tuple[SamplerState, ThermodynamicState, Optional[PairsBase]]:
        """
        Performs an individual MC step.

        This will call the _propose  function which will be specific to the type of move.

        Parameters
        ----------
        current_sampler_state : SamplerState, required
            Current sampler state.
        current_thermodynamic_state : ThermodynamicState, required
            Current thermodynamic state.
        current_nbr_list : Optional[PairsBase]
            Neighbor list associated with the current state.

        Returns
        -------
        sampler_state : SamplerState
            The updated sampler state; if a move is rejected this will be unchanged.
            Note, if the proposed move is rejected, the current PRNG key will be updated to ensure
            that we are using a different random number for the next iteration.
        thermodynamic_state : ThermodynamicState
            The updated thermodynamic state; if a move is rejected this will be unchanged.
            Note, many MC moves will not modify the thermodynamic state regardless of acceptance of the move.
        nbr_list: PairsBase, optional
            The updated neighbor/pair list. If a nbr_list is not set, this will be None.
            If the move is rejected, this will correspond to the neighbor

        """

        # if this is the first time we are calling this function during this iteration
        # we will need to calculate the reduced potential for the current state
        # this is toggled by the calculate_current_reduced_potential flag
        # otherwise, we can use the one that was saved from the last step, for efficiency
        if self._current_reduced_potential is None:
            current_reduced_potential = (
                current_thermodynamic_state.get_reduced_potential(
                    current_sampler_state, current_nbr_list
                )
            )
            # save the current_reduced_potential so we don't have to recalculate
            # it on the next iteration if the move is rejected
            self._current_reduced_potential = current_reduced_potential
        else:
            current_reduced_potential = self._current_reduced_potential

        # propose a new state and calculate the log proposal ratio
        # this will be specific to the type of move
        # in addition to the sampler_state, this will require/return the thermodynamic state
        # for systems that e.g., make changes to particle identity.
        # For efficiency, we will also return a copy of the nbr_list associated with the proposed state
        # because if the move is rejected, we can move back the original state without having to rebuild the nbr_list
        # if it were modified due to the proposed state.
        (
            proposed_sampler_state,
            proposed_thermodynamic_state,
            proposed_reduced_potential,
            log_proposal_ratio,
            proposed_nbr_list,
        ) = self._propose(
            current_sampler_state,
            current_thermodynamic_state,
            current_reduced_potential,
            current_nbr_list,
        )

        if jnp.isnan(proposed_reduced_potential):
            decision = False
        else:
            # accept or reject the proposed state
            decision = self._accept_or_reject(
                log_proposal_ratio,
                proposed_sampler_state.new_PRNG_key,
                acceptance_method=self.acceptance_method,
            )
        # a function that will update the statistics for the move

        self._update_statistics(decision)

        if decision:
            # save the reduced potential of the accepted state so
            # we don't have to recalculate it the next iteration
            self._current_reduced_potential = proposed_reduced_potential

            # replace the current state with the proposed state
            # not sure this needs to be a separate function but for simplicity in outlining the code it is fine
            # or should this return the new sampler_state and thermodynamic_state?

            return (
                proposed_sampler_state,
                proposed_thermodynamic_state,
                proposed_nbr_list,
            )
        else:
            # if we reject the move, we need to update the current_PRNG key to ensure that
            # we are using a different random number for the next iteration
            # this is needed because the _step function returns a SamplerState instead of updating it in place
            current_sampler_state._current_PRNG_key = (
                proposed_sampler_state._current_PRNG_key
            )

            return current_sampler_state, current_thermodynamic_state, current_nbr_list

    def _update_statistics(self, decision):
        """
        Update the statistics for the move.
        """
        if decision:
            self.n_accepted += 1
        self.n_proposed += 1

    @property
    def statistics(self):
        """The acceptance statistics as a dictionary."""
        return dict(n_accepted=self.n_accepted, n_proposed=self.n_proposed)

    @statistics.setter
    def statistics(self, value):
        self.n_accepted = value["n_accepted"]
        self.n_proposed = value["n_proposed"]

    def reset_statistics(self):
        """Reset the acceptance statistics."""
        self.n_accepted = 0
        self.n_proposed = 0

    @abstractmethod
    def _propose(
        self,
        current_sampler_state: SamplerState,
        current_thermodynamic_state: ThermodynamicState,
        current_reduced_potential: float,
        current_nbr_list: Optional[PairsBase] = None,
    ) -> Tuple[SamplerState, ThermodynamicState, float, float, Optional[PairsBase]]:
        """
        Propose a new state and calculate the log proposal ratio.

        This will accept the relevant quantities for the current state, returning the proposed state quantities
        and the log proposal ratio.

        This will need to be defined for each new move.

        Parameters
        ----------
        current_sampler_state : SamplerState, required
            Current sampler state.
        current_thermodynamic_state : ThermodynamicState, required
            Current thermodynamic state.
        current_reduced_potential : float, required
            Current reduced potential.
        current_nbr_list : PairsBase, required
            Neighbor list associated with the current state.

        Returns
        -------
        proposed_sampler_state : SamplerState
            Proposed sampler state.
        proposed_thermodynamic_state : ThermodynamicState
            Proposed thermodynamic state.
        proposed_reduced_potential : float
            Proposed reduced potential.
        log_proposal_ratio : float
            Log proposal ratio.
        proposed_nbr_list : PairsBase
            Proposed neighbor list. If not defined, this will be None.

        """
        pass

    def _accept_or_reject(
        self,
        log_proposal_ratio,
        key,
        acceptance_method,
    ):
        """
        Accept or reject the proposed state with a given methodology.
        """
        # define the acceptance probability
        if acceptance_method == "Metropolis-Hastings":
            import jax.random as jrandom

            compare_to = jrandom.uniform(key)
            if -log_proposal_ratio <= 0.0 or compare_to < jnp.exp(log_proposal_ratio):
                return True
            else:
                return False


class MonteCarloDisplacementMove(MCMove):
    """
    A Monte Carlo move that randomly displaces particles in the system.

    For each move, all particles will be randomly displaced at once, where the random displacement is drawn from
    a normal distribution. The standard deviation of the distribution is defined by the `displacement_sigma` parameter.

    Displacements can be restricted to a subset of particles by defining the `atom_subset` parameter, which is a list of
    particle indices that will be allowed to move. If `atom_subset` is not defined, all particles will be displaced.

    Note, the displacement moves are applied on a per-particle basis; this does not support collective moves.

    The value of the `displacement_sigma` can be autotuned to achieve a target acceptance ratio between 0.4 and 0.6,
    by setting the autotune parameter to True. The frequency of autotuning is defined by setting `autotune_interval`.


    """

    def __init__(
        self,
        displacement_sigma=1.0 * unit.nanometer,
        number_of_moves: int = 100,
        atom_subset: Optional[List[int]] = None,
        report_interval: int = 1,
        reporter: Optional[MCReporter] = None,
        autotune: bool = False,
        autotune_interval: int = 100,
        acceptance_method="Metropolis-Hastings",
    ):
        """
        Initialize the Displacement Move class.

        Parameters
        ----------
        displacement_sigma : float or unit.Quantity, optional
            The standard deviation of the displacement for each move. Default is 1.0 nm.
        number_of_moves : int, optional
            The number of move attempts to perform. Default is 100.
            For a given move, all particles will  be randomly displaced at once (unless atom_subset is),
            rather than moving each particle one at a time.
        atom_subset : list of int, optional
            A list of particle indices that represent a subset of all particles.
            If defined, only those particles in the list will have their positions random displaced.
            Default is None.
        reporter : SimulationReporter, optional
            The reporter to write the data to. Default is None.
        autotune : bool, optional
            Whether to autotune the displacement_sigma of the move to achieve an acceptance ratio between 0.4 and 0.6.
            Default is False.
        autotune_interval : int, optional
            Frequency of autotuning displacement_sigma of the move. Default is 100.
        acceptance_method : str, optional
            Methodology to use for accepting or rejecting the proposed state.
            Default is "Metropolis-Hastings".

        Returns
        -------
        None
        """
        super().__init__(
            number_of_moves=number_of_moves,
            reporter=reporter,
            report_interval=report_interval,
            autotune=autotune,
            autotune_interval=autotune_interval,
            acceptance_method=acceptance_method,
        )
        self.displacement_sigma = displacement_sigma

        self.atom_subset = atom_subset
        self.atom_subset_mask = None

    def _report(
        self,
        step: int,
        iteration: int,
        number_of_attempts_made: int,
        acceptance_probability: float,
        sampler_state: SamplerState,
        thermodynamic_state: ThermodynamicState,
        nbr_list: Optional[PairsBase] = None,
    ):
        """
        Report the current state of the MC displacement move.

        Parameters
        ----------
        step : int
            The current step of the simulation move.
        iteration : int
            The current iteration of the move sequence (i.e., how many times has this been called thus far).
        number_of_attempts_made : int
            The total number of steps that have been taken in the simulation move. step+ nr_moves*iteration
        acceptance_probability : float
            The acceptance probability of the move.
        sampler_state : SamplerState
            The sampler state of the system.
        thermodynamic_state : ThermodynamicState
            The thermodynamic state of the system.
        nbr_list : Optional[PairBase]=None
            The neighbor list or pair list for evaluating interactions in the system, default None

        """
        potential = thermodynamic_state.potential.compute_energy(
            sampler_state.positions, nbr_list
        )
        self.reporter.report(
            {
                "step": step,
                "iteration": iteration,
                "number_of_attempts_made": number_of_attempts_made,
                "potential_energy": potential,
                "displacement_sigma": self.displacement_sigma.value_in_unit_system(
                    unit.md_unit_system
                ),
                "acceptance_probability": acceptance_probability,
            }
        )

    def _autotune(self):
        """
        Update the displacement_sigma to reach a target acceptance probability between 0.4 and 0.6.
        """
        acceptance_ratio = self.n_accepted / self.n_proposed
        if acceptance_ratio > 0.6:
            self.displacement_sigma *= 1.1
        elif acceptance_ratio < 0.4:
            self.displacement_sigma /= 1.1

    def _propose(
        self,
        current_sampler_state: SamplerState,
        current_thermodynamic_state: ThermodynamicState,
        current_reduced_potential: float,
        current_nbr_list: Optional[PairsBase] = None,
    ) -> Tuple[SamplerState, ThermodynamicState, float, float, Optional[PairsBase]]:
        """
        Implements the logic specific to displacement moves.

        Parameters
        ----------
        current_sampler_state : SamplerState, required
            Current sampler state.
        current_thermodynamic_state : ThermodynamicState, required
            Current thermodynamic state.
        current_reduced_potential : float, required
            Current reduced potential.
        current_nbr_list : Optional[PairsBase]
            Neighbor list associated with the current state.

        Returns
        -------
        proposed_sampler_state : SamplerState
            Proposed sampler state.
        proposed_thermodynamic_state : ThermodynamicState
            Proposed thermodynamic state.
        proposed_reduced_potential : float
            Proposed reduced potential.
        log_proposal_ratio : float
            Log proposal ratio.
        proposed_nbr_list : PairsBase
            Proposed neighbor list. If not defined, this will be None.
        """

        # create a mask for the atom subset: if a value of the mask is 0
        # the particle won't move; if 1 the particle will be moved
        if self.atom_subset is not None and self.atom_subset_mask is None:
            import jax.numpy as jnp

            self.atom_subset_mask = jnp.zeros(current_sampler_state.number_of_particles)
            for atom in self.atom_subset:
                self.atom_subset_mask = self.atom_subset_mask.at[atom].set(1)

        key = current_sampler_state.new_PRNG_key

        nr_of_atoms = current_sampler_state.number_of_particles

        unitless_displacement_sigma = self.displacement_sigma.value_in_unit_system(
            unit.md_unit_system
        )
        import jax.random as jrandom

        scaled_displacement_vector = (
            jrandom.normal(key, shape=(nr_of_atoms, 3)) * unitless_displacement_sigma
        )
        import copy

        proposed_sampler_state = copy.deepcopy(current_sampler_state)

        if self.atom_subset is not None:
            proposed_sampler_state.positions = (
                proposed_sampler_state.positions
                + scaled_displacement_vector * self.atom_subset_mask
            )
        else:
            proposed_sampler_state.positions = (
                proposed_sampler_state.positions + scaled_displacement_vector
            )

        # after proposing a move we need to wrap particles and see if we need to rebuild the neighborlist
        if current_nbr_list is not None:
            proposed_sampler_state.positions = current_nbr_list.space.wrap(
                proposed_sampler_state.positions,
                proposed_sampler_state.box_vectors,
            )

            # if we need to rebuild the neighbor the neighborlist
            # we will make a copy and then build
            if current_nbr_list.check(proposed_sampler_state.positions):
                import copy

                proposed_nbr_list = copy.deepcopy(current_nbr_list)

                proposed_nbr_list.build(
                    proposed_sampler_state.positions, proposed_sampler_state.box_vectors
                )
            # if we don't need to update the neighborlist, just make a new variable that refers to the original
            else:
                proposed_nbr_list = current_nbr_list
        else:
            proposed_nbr_list = None

        proposed_reduced_potential = current_thermodynamic_state.get_reduced_potential(
            proposed_sampler_state, proposed_nbr_list
        )

        log_proposal_ratio = -proposed_reduced_potential + current_reduced_potential

        # since do not change the thermodynamic state we can return
        # 'current_thermodynamic_state' rather than making a copy
        return (
            proposed_sampler_state,
            current_thermodynamic_state,
            proposed_reduced_potential,
            log_proposal_ratio,
            proposed_nbr_list,
        )


class MonteCarloBarostatMove(MCMove):
    """
    A Monte Carlo move that randomly changes the volume of the system.

    The volume change is drawn from a normal distribution with a mean of 0 and a standard deviation defined
    by the product of the `volume_max_scale` parameter and the current volume. Particle positions are scaled
    proportionately with the change in volume. This routine operates on a per-particle basis and does not support
    collective moves (i.e., it is an "atomic" barostat move where particle center-of-mass positions are scaled;
    it is not aware of "molecules" which would be scaled by the molecule center-of-mass).

    The `volume_max_scale` parameter can be autotuned to achieve a target acceptance ratio between 0.25 and 0.75,
    by setting the autotune parameter to True. The frequency of autotuning is defined by setting `autotune_interval`.
    Note, the maximum value of `volume_max_scale` is capped at 0.3 in the auto-tuning process.


    """

    def __init__(
        self,
        volume_max_scale=0.01,
        number_of_moves: int = 100,
        report_interval: int = 1,
        reporter: Optional[LangevinDynamicsReporter] = None,
        autotune: bool = False,
        autotune_interval: int = 100,
        acceptance_method="Metropolis-Hastings",
    ):
        """
        Initialize the Monte Carlo Barostat Move class.

        Parameters
        ----------
        volume_max_scale : float, optional
            The scaling factor multiplied by volume to set the maximum volume change allowed.
        number_of_moves : int, optional
            The number of volume update moves attempts to perform. Default is 100.
        reporter : SimulationReporter, optional
            The reporter to write the data to. Default is None.
        autotune : bool, optional
            Whether to autotune the volume_max_scale value of the move to achieve a target probability
            between 0.25 and 0.75. Default is False. volume_max_scale is capped at 0.3
        autotune_interval : int, optional
            Frequency of autotuning the volume_max_scale of the move. Default is 100.
        acceptance_method : str, optional
            Methodology to use for accepting or rejecting the proposed state.
            Default is "Metropolis-Hastings".

        Returns
        -------
        None
        """
        super().__init__(
            number_of_moves=number_of_moves,
            reporter=reporter,
            report_interval=report_interval,
            autotune=autotune,
            autotune_interval=autotune_interval,
            acceptance_method=acceptance_method,
        )
        self.volume_max_scale = volume_max_scale

    def _report(
        self,
        step: int,
        iteration: int,
        number_of_attempts_made: int,
        acceptance_probability: float,
        sampler_state: SamplerState,
        thermodynamic_state: ThermodynamicState,
        nbr_list: Optional[PairsBase] = None,
    ):
        """

        Parameters
        ----------
        step : int
            The current step of the simulation move.
        iteration : int
            The current iteration of the move sequence (i.e., how many times has this been called thus far).
        number_of_attempts_made : int
            The total number of steps that have been taken in the simulation move. step+ nr_moves*iteration
        acceptance_probability : float
            The acceptance probability of the move.
        sampler_state : SamplerState
            The sampler state of the system.
        thermodynamic_state : ThermodynamicState
            The thermodynamic state of the system.
        nbr_list : Optional[PairBase]=None
            The neighbor list or pair list for evaluating interactions in the system, default None
        """

        potential = thermodynamic_state.potential.compute_energy(
            sampler_state.positions, nbr_list
        )
        volume = (
            sampler_state.box_vectors[0][0]
            * sampler_state.box_vectors[1][1]
            * sampler_state.box_vectors[2][2]
        )
        self.reporter.report(
            {
                "step": step,
                "iteration": iteration,
                "number_of_attempts_made": number_of_attempts_made,
                "potential_energy": potential,
                "volume": volume,
                "box_vectors": sampler_state.box_vectors,
                "max_volume_scale": self.volume_max_scale,
                "acceptance_probability": acceptance_probability,
            }
        )

    def _autotune(self):
        """
        Update the volume_max_scale parameter to ensure our acceptance probability is within the range of 0.25 to 0.75.
        The maximum volume_max_scale will be capped at 0.3.
        """
        acceptance_ratio = self.n_accepted / self.n_proposed
        if acceptance_ratio < 0.25:
            self.volume_max_scale /= 1.1
        elif acceptance_ratio > 0.75:
            self.volume_max_scale = min(self.volume_max_scale * 1.1, 0.3)

    def _propose(
        self,
        current_sampler_state: SamplerState,
        current_thermodynamic_state: ThermodynamicState,
        current_reduced_potential: float,
        current_nbr_list: Optional[PairsBase] = None,
    ) -> Tuple[SamplerState, ThermodynamicState, float, float, Optional[PairsBase]]:
        """
        Implement the logic specific to displacement changes.

        Parameters
        ----------
        current_sampler_state : SamplerState, required
            Current sampler state.
        current_thermodynamic_state : ThermodynamicState, required
            Current thermodynamic state.
        current_reduced_potential : float, required
            Current reduced potential.
        current_nbr_list : PairsBase, optional
            Neighbor list associated with the current state.

        Returns
        -------
        proposed_sampler_state : SamplerState
            Proposed sampler state.
        proposed_thermodynamic_state : ThermodynamicState
            Proposed thermodynamic state.
        proposed_reduced_potential : float
            Proposed reduced potential.
        log_proposal_ratio : float
            Log proposal ratio.
        proposed_nbr_list : PairsBase
            Proposed neighbor list. If not defined, this will be None.

        """
        from loguru import logger as log

        key = current_sampler_state.new_PRNG_key

        import jax.random as jrandom

        nr_of_atoms = current_sampler_state.number_of_particles

        initial_volume = (
            current_sampler_state.box_vectors[0][0]
            * current_sampler_state.box_vectors[1][1]
            * current_sampler_state.box_vectors[2][2]
        )

        # Calculate the maximum amount the volume can change by
        delta_volume_max = self.volume_max_scale * initial_volume

        # Calculate the volume change by generating a random number between -1 and 1
        # and multiplying by the maximum allowed volume change, delta_volume_max
        delta_volume = jrandom.uniform(key, minval=-1, maxval=1) * delta_volume_max
        # calculate the new volume
        proposed_volume = initial_volume + delta_volume

        # calculate the length scale factor for particle positions and box vectors
        length_scaling_factor = jnp.power(proposed_volume / initial_volume, 1.0 / 3.0)

        import copy

        proposed_sampler_state = copy.deepcopy(current_sampler_state)
        proposed_sampler_state.positions = (
            current_sampler_state.positions * length_scaling_factor
        )

        proposed_sampler_state.box_vectors = (
            current_sampler_state.box_vectors * length_scaling_factor
        )

        if current_nbr_list is not None:
            proposed_nbr_list = copy.deepcopy(current_nbr_list)
            # after scaling the box vectors and positions we should always rebuild the neighborlist
            proposed_nbr_list.build(
                proposed_sampler_state.positions, proposed_sampler_state.box_vectors
            )

        proposed_reduced_potential = current_thermodynamic_state.get_reduced_potential(
            proposed_sampler_state, proposed_nbr_list
        )
        # NPT acceptance criteria was originally defined in McDonald 1972, https://doi.org/10.1080/00268977200100031
        # (see equation 9). The acceptance probability is given by:
        # ⎡−β (ΔU + PΔV ) + N ln(V new /V old )⎤
        log_proposal_ratio = -(
            proposed_reduced_potential - current_reduced_potential
        ) + nr_of_atoms * jnp.log(proposed_volume / initial_volume)

        # we do not change the thermodynamic state so we can return 'current_thermodynamic_state'
        return (
            proposed_sampler_state,
            current_thermodynamic_state,
            proposed_reduced_potential,
            log_proposal_ratio,
            proposed_nbr_list,
        )


class RotamerMove(MCMove):
    def _propose(self):
        """
        Implement the logic specific to rotamer changes.
        """
        pass


class ProtonationStateMove(MCMove):
    def _propose(self):
        """
        Implement the logic specific to protonation state changes.
        """
        pass


class TautomericStateMove(MCMove):
    def _propose(self):
        """
        Implement the logic specific to tautomeric state changes.
        """
        pass


class MoveSchedule:
    """
    Represents an (optimizable) series of moves for a Markov Chain Monte Carlo (MCMC) algorithm.

    Parameters
    ----------
    move_schedule : List[Tuple[str, StateUpdateMove]]
        A list representing the move schedule, where each tuple contains a move name and a move instance.

    Raises
    ------
    ValueError
        If a move in the schedule is not an instance of StateUpdateMove.
    """

    def __init__(
        self,
        move_schedule: List[Tuple[str, MCMCMove]],
    ) -> None:
        _AVAILABLE_MOVES = ["LangevinDynamicsMove"]
        self.move_schedule = move_schedule

        self._validate_sequence()

    def _validate_sequence(self):
        """
        Validates the move sequence against the available moves.

        Raises
        ------
        ValueError
            If a move in the sequence is not present in available_moves.
        """
        for move_name, move_class in self.move_schedule:
            if not isinstance(move_class, MCMCMove):
                raise ValueError(f"Move {move_name} in the sequence is not available.")


class MCMCSampler:
    """
    Basic Markov chain Monte Carlo sampler.

    Parameters
    ----------
    move_set : MoveSet
        Set of moves to attempt during MCMC run.
    sampler_state : SamplerState
        Initial sampler state.
    thermodynamic_state : ThermodynamicState
        Thermodynamic state describing the system.

    Examples
    """

    def __init__(
        self,
        move_set: MoveSchedule,
    ):
        from loguru import logger as log

        log.info("Initializing MCMC sampler")
        self.move = move_set

    def run(
        self,
        sampler_state: SamplerState,
        thermodynamic_state: ThermodynamicState,
        n_iterations: int = 1,
        nbr_list: Optional[PairsBase] = None,
    ):
        """
        Run the sampler for a specified number of iterations.

        Parameters
        ----------
        sampler_state : SamplerState
            The initial state of the sampler.
        thermodynamic_state : ThermodynamicState
            The thermodynamic state of the system.
        n_iterations : int, optional
            Number of iterations of the sampler to run.
            Default is 1.
        nbr_list : PairsBase, optional
            The neighbor list to use for the simulation.

        Returns
        -------
        sampler_state : SamplerState
            The updated sampler state.
        thermodynamic_state : ThermodynamicState
            The updated thermodynamic state.
        nbr_list: PairsBase
            The updated neighbor/pair list. If a nbr_list is not set, this will be None.

        """
        from loguru import logger as log
        from copy import deepcopy

        sampler_state = deepcopy(sampler_state)
        thermodynamic_state = deepcopy(thermodynamic_state)
        nbr_list = deepcopy(nbr_list)

        log.info("Running MCMC sampler")
        log.info(f"move_schedule = {self.move.move_schedule}")
        for iteration in range(n_iterations):
            log.info(f"Iteration {iteration + 1}/{n_iterations}")
            for move_name, move in self.move.move_schedule:
                log.debug(f"Performing: {move_name}")

                sampler_state, thermodynamic_state, nbr_list = move.update(
                    sampler_state, thermodynamic_state, nbr_list
                )

        log.info("Finished running MCMC sampler")
        log.debug("Closing reporter")
        for _, move in self.move.move_schedule:
            if move.reporter is not None:
                move.reporter.flush_buffer()
                log.debug(f"Closed reporter {move.reporter.log_file_path}")
        return sampler_state, thermodynamic_state, nbr_list
