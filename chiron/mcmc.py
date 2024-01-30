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
        nr_of_moves: int,
        reporter: Optional[_SimulationReporter] = None,
        report_frequency: Optional[int] = 100,
    ):
        """
        Initialize a move within the molecular system.

        Parameters
        ----------
        nr_of_moves : int
            Number of moves to be applied.
        reporter : _SimulationReporter, optional
            Reporter object for saving the simulation data.
            Default is None.
        report_frequency : int, optional
            Frequency of saving the simulation data in the reporter.
            Default is 100.

        """

        self.nr_of_moves = nr_of_moves
        self.reporter = reporter
        self.report_frequency = report_frequency

        # we need to keep track of which iteration we are on
        self.iteration = 0

        from loguru import logger as log

        if self.reporter is not None:
            log.info(
                f"Using reporter {self.reporter} saving to {self.reporter.workdir}"
            )
            assert self.report_frequency is not None

    @abstractmethod
    def update(
        self,
        sampler_state: SamplerState,
        thermodynamic_state: ThermodynamicState,
        nbr_list: Optional[PairsBase] = None,
    ):
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

        """
        pass


class LangevinDynamicsMove(MCMCMove):
    def __init__(
        self,
        stepsize=1.0 * unit.femtoseconds,
        collision_rate=1.0 / unit.picoseconds,
        initialize_velocities: bool = False,
        reinitialize_velocities: bool = False,
        reporter: Optional[LangevinDynamicsReporter] = None,
        report_frequency: int = 100,
        nr_of_steps=1_000,
        save_traj_in_memory: bool = False,
    ):
        """
        Initialize the LangevinDynamicsMove with a molecular system.

        Parameters
        ----------
        stepsize : unit.Quantity
            Time step size for the integration.
        collision_rate : unit.Quantity
            Collision rate for the Langevin dynamics.
        initialize_velocities: bool, optional
            Whether to initialize the velocities the first time the run function is called.
            Default is False.
        reinitialize_velocities : bool, optional
            Whether to reinitialize the velocities each time the run function is called.
            Default is False.
        reporter : LangevinDynamicsReporter, optional
            Reporter object for saving the simulation data.
            Default is None.
        report_frequency : int
            Frequency of saving the simulation data.
            Default is 100.
        nr_of_steps : int, optional
            Number of steps to run the integrator for.
            Default is 1_000.
        save_traj_in_memory: bool
            Flag indicating whether to save the trajectory in memory.
            Default is False. NOTE: Only for debugging purposes.
        """
        super().__init__(
            nr_of_moves=nr_of_steps,
            reporter=reporter,
            report_frequency=report_frequency,
        )

        self.stepsize = stepsize
        self.collision_rate = collision_rate
        self.save_traj_in_memory = save_traj_in_memory
        self.traj = []
        from chiron.integrators import LangevinIntegrator

        self.integrator = LangevinIntegrator(
            stepsize=self.stepsize,
            collision_rate=self.collision_rate,
            initialize_velocities=initialize_velocities,
            reinitialize_velocities=reinitialize_velocities,
            report_frequency=report_frequency,
            reporter=reporter,
            save_traj_in_memory=save_traj_in_memory,
        )

    def update(
        self,
        sampler_state: SamplerState,
        thermodynamic_state: ThermodynamicState,
        nbr_list: Optional[PairsBase] = None,
    ):
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

        """

        assert isinstance(
            sampler_state, SamplerState
        ), f"Sampler state must be SamplerState, not {type(sampler_state)}"
        assert isinstance(
            thermodynamic_state, ThermodynamicState
        ), f"Thermodynamic state must be ThermodynamicState, not {type(thermodynamic_state)}"

        sampler_state = self.integrator.run(
            thermodynamic_state=thermodynamic_state,
            sampler_state=sampler_state,
            n_steps=self.nr_of_moves,
            nbr_list=nbr_list,
        )

        if self.save_traj_in_memory:
            self.traj.append(self.integrator.traj)
            self.integrator.traj = []

        self.iteration += 1

        # The thermodynamic_state will not change for the langevin move
        return sampler_state, thermodynamic_state


class MCMove(MCMCMove):
    def __init__(
        self,
        nr_of_moves: int,
        reporter: Optional[_SimulationReporter],
        report_frequency: int = 1,
        update_stepsize: bool = False,
        update_stepsize_frequency: int = 100,
        method: str = "metropolis",
    ) -> None:
        """
        Initialize the move.

        Parameters
        ----------
        nr_of_moves
            Number of moves to be applied in each call to update.
        reporter
            Reporter object for saving the simulation step data.
        report_frequency
            Frequency of saving the simulation data.
        update_stepsize
            Whether to update the "stepsize" of the move. Stepsize is a generic term for the key move parameters.
            For example, for a simple displacement move this would be the displacement_sigma.
        update_stepsize_frequency
            Frequency of updating the stepsize of the move.
        method
            Methodology to use for accepting or rejecting the proposed state.
            Default is "metropolis".
        """
        super().__init__(
            nr_of_moves,
            reporter=reporter,
            report_frequency=report_frequency,
        )
        self.method = method  # I think we should pass a class/function instead of a string, like space.

        self.reset_statistics()
        self.update_stepsize = update_stepsize
        self.update_stepsize_frequency = update_stepsize_frequency

    def update(
        self,
        sampler_state: SamplerState,
        thermodynamic_state: ThermodynamicState,
        nbr_list: Optional[PairsBase] = None,
    ):
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
        """
        calculate_current_potential = True

        for i in range(self.nr_of_moves):
            sampler_state, thermodynamic_state = self._step(
                sampler_state,
                thermodynamic_state,
                nbr_list,
                calculate_current_potential=calculate_current_potential,
            )
            # after the first step, we don't need to recalculate the current potential, it will be stored
            calculate_current_potential = False

            elapsed_step = i + self.iteration * self.nr_of_moves
            if hasattr(self, "reporter"):
                if self.reporter is not None:
                    # I think it makes sense to i + self.nr_of_moves*self.iteration as our current "step"
                    # otherwise, instances where self.report_frequency > self.nr_of_moves would only report on the
                    # first step which might actually be more frequent than we specify

                    if elapsed_step % self.report_frequency == 0:
                        self._report(
                            i,
                            self.iteration,
                            self.n_accepted / self.n_proposed,
                            sampler_state,
                            thermodynamic_state,
                            nbr_list,
                        )
            if self.update_stepsize:
                # if we only used i,  we might never actually update the parameters if we have a move that is called infrequently
                if (
                    elapsed_step % self.update_stepsize_frequency == 0
                    and elapsed_step > 0
                ):
                    self._update_stepsize()

        self.iteration += 1

        return sampler_state, thermodynamic_state

    @abstractmethod
    def _report(
        self,
        step: int,
        iteration: int,
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
    def _update_stepsize(self):
        """
        Update the "stepsize" for a move to reach a target acceptance probability range.
        This will be specific to the type of move, e.g., a displacement_sigma for a displacement move
        or a maximum volume change factor for a Monte Carlo barostat move.

        Since different moves will be modifying different quantities, this needs to be defined for each move.
        """
        pass

    def _step(
        self,
        current_sampler_state,
        current_thermodynamic_state,
        nbr_list,
        calculate_current_potential=True,
    ):
        # if this is the first time we are calling this,
        # we will need to recalculate the reduced potential for the current state
        # this is toggled by the calculate_current_potential flag
        if calculate_current_potential:
            current_reduced_pot = current_thermodynamic_state.get_reduced_potential(
                current_sampler_state, nbr_list
            )
            # save the current_reduced_pot so we don't have to recalculate
            # it on the next iteration if the move is rejected
            self._current_reduced_pot = current_reduced_pot
        else:
            current_reduced_pot = self._current_reduced_pot

        # propose a new state and calculate the log proposal ratio
        # this will be specific to the type of move
        # in addition to the sampler_state, this will require/return the thermodynamic state
        # for systems that e.g., make changes to particle identity.
        (
            proposed_sampler_state,
            proposed_thermodynamic_state,
            proposed_reduced_pot,
            log_proposal_ratio,
        ) = self._propose(
            current_sampler_state,
            current_thermodynamic_state,
            current_reduced_pot,
            nbr_list,
        )

        # accept or reject the proposed state
        decision = self._accept_or_reject(
            log_proposal_ratio,
            proposed_sampler_state.new_PRNG_key,
            method=self.method,
        )
        # a function that will update the statistics for the move

        if jnp.isnan(proposed_reduced_pot):
            decision = False

        self._update_statistics(decision)

        if decision:
            # save the reduced potential of the accepted state so
            # we don't have to recalculate it the next iteration
            self._current_reduced_pot = proposed_reduced_pot

            # replace the current state with the proposed state
            # not sure this needs to be a separate function but for simplicity in outlining the code it is fine
            # or should this return the new sampler_state and thermodynamic_state?

            return proposed_sampler_state, proposed_thermodynamic_state
        else:
            # if we reject the move, we need to update the current_PRNG key to ensure that
            # we are using a different random number for the next iteration
            # this is needed because the _step function returns a SamplerState instead of updating it in place
            current_sampler_state._current_PRNG_key = (
                proposed_sampler_state._current_PRNG_key
            )
            if nbr_list is not None:
                if nbr_list.check(current_sampler_state.x0):
                    nbr_list.build(
                        current_sampler_state.x0, current_sampler_state.box_vectors
                    )

            return current_sampler_state, current_thermodynamic_state

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
        self, current_sampler_state, current_thermodynamic_state, current_reduced_pot
    ):
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
        current_reduced_pot : float, required
            Current reduced potential.

        Returns
        -------
        proposed_sampler_state : SamplerState
            Proposed sampler state.
        proposed_thermodynamic_state : ThermodynamicState
            Proposed thermodynamic state.
        proposed_reduced_pot : float
            Proposed reduced potential.
        log_proposal_ratio : float
            Log proposal ratio.

        """
        pass

    def _accept_or_reject(
        self,
        log_proposal_ratio,
        key,
        method,
    ):
        """
        Accept or reject the proposed state with a given methodology.
        """
        # define the acceptance probability
        if method == "metropolis":
            import jax.random as jrandom

            compare_to = jrandom.uniform(key)
            if -log_proposal_ratio <= 0.0 or compare_to < jnp.exp(log_proposal_ratio):
                return True
            else:
                return False


class MetropolisDisplacementMove(MCMove):
    def __init__(
        self,
        displacement_sigma=1.0 * unit.nanometer,
        nr_of_moves: int = 100,
        atom_subset: Optional[List[int]] = None,
        report_frequency: int = 1,
        reporter: Optional[LangevinDynamicsReporter] = None,
        update_stepsize: bool = True,
        update_stepsize_frequency: int = 100,
    ):
        """
        Initialize the Displacement Move class.

        Parameters
        ----------
        displacement_sigma : float or unit.Quantity, optional
            The standard deviation of the displacement for each move. Default is 1.0 nm.
        nr_of_moves : int, optional
            The number of moves to perform. Default is 100.
        atom_subset : list of int, optional
            A subset of atom indices to consider for the moves. Default is None.
        reporter : SimulationReporter, optional
            The reporter to write the data to. Default is None.
        update_stepsize : bool, optional
            Whether to update the stepsize of the move. Default is True.
        update_stepsize_frequency : int, optional
            Frequency of updating the stepsize of the move. Default is 100.
        Returns
        -------
        None
        """
        super().__init__(
            nr_of_moves=nr_of_moves,
            reporter=reporter,
            report_frequency=report_frequency,
            update_stepsize=update_stepsize,
            update_stepsize_frequency=update_stepsize_frequency,
            method="metropolis",
        )
        self.displacement_sigma = displacement_sigma

        self.atom_subset = atom_subset
        self.atom_subset_mask = None

    def _report(
        self,
        step,
        iteration,
        acceptance_probability,
        sampler_state,
        thermodynamic_state,
        nbr_list,
    ):
        """
        Report the current state of the MC displacement move.

        Parameters
        ----------
        step : int
            The current step of the simulation move.
        iteration : int
            The current iteration of the move sequence (i.e., how many times has this been called thus far).
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
            sampler_state.x0, nbr_list
        )
        self.reporter.report(
            {
                "step": step,
                "iteration": iteration,
                "potential_energy": potential,
                "displacement_sigma": self.displacement_sigma.value_in_unit_system(
                    unit.md_unit_system
                ),
                "acceptance_probability": acceptance_probability,
            }
        )

    def _update_stepsize(self):
        """
        Update the displacement_sigma to reach a target acceptance probability of 0.5.
        """
        acceptance_ratio = self.n_accepted / self.n_proposed
        if acceptance_ratio > 0.6:
            self.displacement_sigma *= 1.1
        elif acceptance_ratio < 0.4:
            self.displacement_sigma /= 1.1

    def _propose(
        self,
        current_sampler_state,
        current_thermodynamic_state,
        current_reduced_pot,
        nbr_list,
    ):
        """
        Implement the logic specific to displacement changes.
        """
        if self.atom_subset is not None and self.atom_subset_mask is None:
            import jax.numpy as jnp

            self.atom_subset_mask = jnp.zeros(current_sampler_state.n_particles)
            for atom in self.atom_subset:
                self.atom_subset_mask = self.atom_subset_mask.at[atom].set(1)

        key = current_sampler_state.new_PRNG_key

        nr_of_atoms = current_sampler_state.n_particles

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
            proposed_sampler_state.x0 = (
                proposed_sampler_state.x0
                + scaled_displacement_vector * self.atom_subset_mask
            )
        else:
            proposed_sampler_state.x0 = (
                proposed_sampler_state.x0 + scaled_displacement_vector
            )

        # after proposing a move we need to wrap particles and see if we need to rebuild
        # the neighborlist
        if nbr_list is not None:
            proposed_sampler_state.x0 = nbr_list.space.wrap(proposed_sampler_state.x0)

            if nbr_list.check(proposed_sampler_state.x0):
                nbr_list.build(
                    proposed_sampler_state.x0, proposed_sampler_state.box_vectors
                )

        proposed_reduced_pot = current_thermodynamic_state.get_reduced_potential(
            proposed_sampler_state, nbr_list
        )

        log_proposal_ratio = -proposed_reduced_pot + current_reduced_pot

        # since do not change the thermodynamic state we can return
        # 'current_thermodynamic_state' rather than making a copy
        return (
            proposed_sampler_state,
            current_thermodynamic_state,
            proposed_reduced_pot,
            log_proposal_ratio,
        )


class MonteCarloBarostatMove(MCMove):
    def __init__(
        self,
        volume_max_scale=0.01,
        nr_of_moves: int = 100,
        atom_subset: Optional[List[int]] = None,
        report_frequency: int = 1,
        reporter: Optional[LangevinDynamicsReporter] = None,
        update_stepsize: bool = True,
        update_stepsize_frequency: int = 100,
    ):
        """
        Initialize the Monte Carlo Barostat Move class.

        Parameters
        ----------
        displacement_sigma : float or unit.Quantity, optional
            The standard deviation of the displacement for each move. Default is 1.0 nm.
        nr_of_moves : int, optional
            The number of moves to perform. Default is 100.
        atom_subset : list of int, optional
            A subset of atom indices to consider for the moves. Default is None.
        reporter : SimulationReporter, optional
            The reporter to write the data to. Default is None.
        update_stepsize : bool, optional
            Whether to update the stepsize of the move. Default is True.
        update_stepsize_frequency : int, optional
            Frequency of updating the stepsize of the move. Default is 100.
        Returns
        -------
        None
        """
        super().__init__(
            nr_of_moves=nr_of_moves,
            reporter=reporter,
            report_frequency=report_frequency,
            update_stepsize=update_stepsize,
            update_stepsize_frequency=update_stepsize_frequency,
            method="metropolis",
        )
        self.volume_max_scale = volume_max_scale

    def _report(
        self,
        step,
        iteration,
        acceptance_probability,
        sampler_state,
        thermodynamic_state,
        nbr_list,
    ):
        """

        Parameters
        ----------
        step : int
            The current step of the simulation move.
        iteration : int
            The current iteration of the move sequence (i.e., how many times has this been called thus far).
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
            sampler_state.x0, nbr_list
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
                "potential_energy": potential,
                "volume": volume,
                "max_volume_scale": self.volume_max_scale,
                "acceptance_probability": acceptance_probability,
            }
        )

    def _update_stepsize(self):
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
        current_sampler_state,
        current_thermodynamic_state,
        current_reduced_pot,
        nbr_list,
    ):
        """
        Implement the logic specific to displacement changes.
        """
        from loguru import logger as log

        key = current_sampler_state.new_PRNG_key

        import jax.random as jrandom

        nr_of_atoms = current_sampler_state.n_particles

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
        proposed_sampler_state.x0 = current_sampler_state.x0 * length_scaling_factor

        proposed_sampler_state.box_vectors = (
            current_sampler_state.box_vectors * length_scaling_factor
        )

        if nbr_list is not None:
            # after scaling the box vectors we should rebuild the neighborlist
            nbr_list.build(
                proposed_sampler_state.x0, proposed_sampler_state.box_vectors
            )

        proposed_reduced_pot = current_thermodynamic_state.get_reduced_potential(
            proposed_sampler_state, nbr_list
        )

        #  χ = exp ⎡−β (ΔU + PΔV ) + N ln(V new /V old )⎤
        log_proposal_ratio = (
            -proposed_reduced_pot
            + current_reduced_pot
            + nr_of_atoms * jnp.log(proposed_volume / initial_volume)
        )

        # we do not change the thermodynamic state so we can return 'current_thermodynamic_state'
        return (
            proposed_sampler_state,
            current_thermodynamic_state,
            proposed_reduced_pot,
            log_proposal_ratio,
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
        sampler_state: SamplerState,
        thermodynamic_state: ThermodynamicState,
    ):
        from copy import deepcopy
        from loguru import logger as log

        log.info("Initializing Gibbs sampler")
        self.move = move_set
        self.sampler_state = deepcopy(sampler_state)
        self.thermodynamic_state = deepcopy(thermodynamic_state)

    def run(self, n_iterations: int = 1, nbr_list: Optional[PairsBase] = None):
        """
        Run the sampler for a specified number of iterations.

        Parameters
        ----------
        n_iterations : int, optional
            Number of iterations of the sampler to run.
        """
        from loguru import logger as log

        log.info("Running MCMC sampler")
        log.info(f"move_schedule = {self.move.move_schedule}")
        for iteration in range(n_iterations):
            log.info(f"Iteration {iteration + 1}/{n_iterations}")
            for move_name, move in self.move.move_schedule:
                log.debug(f"Performing: {move_name}")
                self.sampler_state, self.thermodynamic_state = move.update(
                    self.sampler_state, self.thermodynamic_state, nbr_list
                )

        log.info("Finished running MCMC sampler")
        log.debug("Closing reporter")
        for _, move in self.move.move_schedule:
            if move.reporter is not None:
                move.reporter.flush_buffer()
                # TODO: flush reporter
                log.debug(f"Closed reporter {move.reporter.log_file_path}")
