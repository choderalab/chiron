# This module implements classes that report simulation data.
from loguru import logger as log

import h5py
import numpy as np

from openmm.app import Topology
from typing import List


class BaseReporter:
    _directory = None

    @classmethod
    def set_directory(cls, directory: str):
        """
        Set the base directory for saving reporter files.

        Parameters
        ----------
        directory : str
            The path to the directory where files will be saved.
        """
        cls._directory = directory

    @classmethod
    def get_directory(cls):
        """
        Get the current directory set for saving reporter files.

        Returns
        -------
        Path
            The path to the directory where files will be saved. Defaults to the
            current working directory if no directory has been set.
        """
        from pathlib import Path

        if cls._directory is None:
            log.debug(
                f"No directory set, using current working directory: {Path.cwd()}"
            )
            return Path.cwd()
        return Path(cls._directory)


class _SimulationReporter:
    def __init__(self, file_name: str, buffer_size: int = 10):
        """
        Initialize the _SimulationReporter class.

        Parameters
        ----------
        file_name : str
            Name of the HDF5 file for writing simulation data.
        buffer_size : int, optional
            The size of the buffer before flushing data to disk (default is 10).
        """
        workdir = BaseReporter.get_directory()
        self.file_path_base = workdir / f"{file_name}"
        self.log_file_path = self.file_path_base.with_suffix(".h5")
        self.workdir = workdir
        self.report_iteration = 0
        import os

        os.makedirs(workdir, exist_ok=True)

        log.info(f"Writing simulation log data to {self.log_file_path}")

        self.buffer_size = buffer_size
        self.buffer = {}

    @property
    def properties_to_report(self):
        return self._default_properties

    @properties_to_report.setter
    def properties_to_report(self, properties: List[str]):
        self._default_properties = properties

    def get_available_keys(self):
        keys = []
        with h5py.File(self.log_file_path, "r") as h5file:
            for key in h5file:
                keys.append(key)
        return keys

    def report(self, data_dict):
        """
        Add new data to the buffer and write the buffer to disk if it's full.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing data to report. Keys are data labels (e.g., 'energy'),
            and values are the data points (usually numpy arrays).
        """
        for key, value in data_dict.items():
            if key not in self.buffer:
                # new key shouldn't trigger a flush
                self.buffer[key] = []
            self.buffer[key].append(value)

        self._flush_buffer_if_necessary()

    def _flush_buffer_if_necessary(self):
        """
        Flush the buffer to disk if it reaches the specified buffer size.
        """
        # NOTE: we assume that every property is updated with the same frequency!
        if all(len(self.buffer[key]) > self.buffer_size for key in self.buffer):
            # flush and reset the buffer
            log.debug(self.buffer)
            self.flush_buffer()

    def _write_to_disk(self, key: str):
        """
        Write buffered data of a given key to the HDF5 file.

        Parameters
        ----------
        key : str
            The key of the data to write to disk.

        """
        log.debug(f"Writing {key} to file")
        if key == "positions" and hasattr(self, "_write_to_trajectory"):
            xyz = np.stack(self.buffer[key])
            self._write_to_trajectory(
                positions=xyz,
            )

        with h5py.File(self.log_file_path, "a") as h5file:
            if key in h5file:
                data = np.array(self.buffer[key])
                dset = h5file[key]
                dset.resize((dset.shape[0] + data.shape[0],) + data.shape[1:])
                dset[-data.shape[0] :] = data
            else:
                data = np.array(self.buffer[key])
                log.debug(f"Creating {key} in {self.log_file_path}")
                h5file.create_dataset(
                    key, data=data, maxshape=(None,) + data.shape[1:], chunks=True
                )

    def reset_reporter_file(self):
        # delete the reporter files
        import os

        # if file exists, delete it
        if os.path.exists(self.log_file_path):
            log.debug(f"Deleting {self.log_file_path}")
            os.remove(self.log_file_path)

    def flush_buffer(self) -> None:
        """
        Write any remaining data in the buffer to disk.

        """
        for key in self.buffer:
            if self.buffer[key]:
                self._write_to_disk(key)
        self._reset_buffer()

    def _reset_buffer(self) -> None:
        """
        Reset the data buffer after writing to disk.
        """
        self.buffer = {key: [] for key in self.buffer}

    def get_property(self, name: str) -> np.ndarray:
        """
        Retrieve a specific property from the HDF5 file.

        Parameters
        ----------
        name : str
            The name of the property to retrieve.

        Returns
        -------
        np.ndarray
            The retrieved property data, if available.
        """
        if name == "positions" and hasattr(self, "read_from_trajectory"):
            return self.read_from_trajectory()

        with h5py.File(self.log_file_path, "r") as h5file:
            if name in h5file:
                data = np.array(h5file[name])
            elif name in self.buffer and name not in h5file:
                data = np.array(self.buffer[name])
            elif name not in h5file:
                log.warning(f"{name} not in HDF5 file")
                return None

            # if name == "u_kn":
            #     return np.transpose(
            #         data, (2, 1, 0)
            #     )  # shape: n_states, n_replicas, n_iterations
            #
            # else:
            return data


from typing import Optional
import mdtraj as md


class MultistateReporter(_SimulationReporter):
    _name = "multistate_reporter"
    _default_properties = [
        "positions",
        "box_vectors",
        "u_kn",
        "state_index",
        "step",
    ]

    def __init__(
        self,
        file_name: Optional[str] = None,
        buffer_size: int = 1,
    ) -> None:
        """
        Initialize the MultistateReporter class.

        Parameters
        ----------
        file_name : Optional[str], optional
            Name of the file for storing multistate simulation data. If None, a
            default name based on the reporter name is used.
        buffer_size : int, optional
            The size of the buffer before flushing data to disk (default is 1).
        """

        if file_name is None:
            file_name = MultistateReporter.get_name()

        super().__init__(file_name=file_name, buffer_size=buffer_size)
        self._replica_reporter = {}

    @classmethod
    def get_name(cls):
        return cls._name

    def _write_to_trajectory(self, positions: np.ndarray) -> None:
        nr_of_frames, n_replicas, n_of_atoms, _ = positions.shape

        for replica_id in range(n_replicas):
            # if file does not exist, create it
            key = f"replica_{replica_id}"
            if self._replica_reporter.get(key) is None:
                self._replica_reporter[key] = LangevinDynamicsReporter(key)

            reporter = self._replica_reporter.get(key)

            for frame_id in range(nr_of_frames):
                data = {"positions": positions[frame_id, replica_id]}
                if self.buffer.get("box_vectors") is not None:
                    data["box_vectors"] = self.buffer.get("box_vectors")[frame_id]
                reporter.report(data)

    def flush_buffer(self):
        for reporter in self._replica_reporter.values():
            reporter.flush_buffer()
            reporter._write_xtc_file_handle.flush()

        return super().flush_buffer()


from typing import Optional


class MCReporter(_SimulationReporter):
    _name = "mc_reporter"

    def __init__(self, file_name: Optional[str] = None, buffer_size: int = 1) -> None:
        """
        Initialize the MCReporter class for Monte Carlo simulations.

        Parameters
        ----------
        file_name : Optional[str], optional
            The file name for storing simulation data.
        buffer_size : int, optional
            The size of the buffer before flushing data to disk.
        """
        if file_name is None:
            file_name = MCReporter.get_name()

        super().__init__(file_name=file_name, buffer_size=buffer_size)

    @classmethod
    def get_name(cls):
        return cls._name


class LangevinDynamicsReporter(_SimulationReporter):
    _name = "langevin_reporter"
    _default_properties = ["positions", "box_vectors", "potential_energy", "step"]

    def __init__(
        self,
        file_name: Optional[str] = None,
        buffer_size: int = 1,
        topology: Optional[Topology] = None,
    ):
        """
        Initialize the LangevinDynamicsReporter for Langevin dynamics simulations.

        Parameters
        ----------
        file_name : Optional[str], optional
            The file name for storing simulation data.
        buffer_size : int, optional
            The size of the buffer before flushing data to disk.
        topology : Optional[Topology], optional
            The system topology for generating trajectories.
        """
        if file_name is None:
            file_name = LangevinDynamicsReporter.get_name()

        super().__init__(file_name=file_name, buffer_size=buffer_size)
        self.topology = topology
        self._write_xtc_file_handle = None
        self.xtc_file_path = f"{self.file_path_base}.xtc"

    @classmethod
    def get_name(cls):
        return cls._name

    def get_mdtraj_trajectory(self) -> md.Trajectory:
        """
        Generate an MDTraj trajectory object from the stored positions.

        Returns
        -------
        md.Trajectory
            The MDTraj trajectory object created from the stored position data.
        """
        import mdtraj as md

        return md.Trajectory(
            xyz=self.get_property("traj"),
            topology=md.Topology.from_openmm(self.topology),
            unitcell_lengths=self.get_property("box_vectors"),
            unitcell_angles=self.get_property("box_angles"),
        )

    def _write_to_trajectory(self, positions: np.ndarray) -> None:
        """
        Write position data to a trajectory file for molecular dynamics.

        Parameters
        ----------
        positions : np.ndarray
            The positions of particles to be written to the trajectory.
        """
        if self._write_xtc_file_handle is None:
            log.debug(f"Creating trajectory in {self.xtc_file_path}")
            self._write_xtc_file_handle = md.formats.XTCTrajectoryFile(
                self.xtc_file_path, mode="w"
            )

        LangevinDynamicsReporter._write_to_xtc(
            file_handler=self._write_xtc_file_handle,
            positions=positions,
            iteration=self.buffer.get("step"),
            box_vectors=self.buffer.get("box_vectors"),
        )

    def read_from_trajectory(self) -> np.ndarray:
        """
        Read position data from a trajectory file.

        Returns
        -------
        np.ndarray
            The positions read from the trajectory file.
        """
        # flush the write buffer
        self._write_xtc_file_handle.flush()
        with md.formats.XTCTrajectoryFile(
            self.xtc_file_path, mode="r"
        ) as _read_xtc_file_handle:
            return LangevinDynamicsReporter._read_from_xtc(_read_xtc_file_handle)

    @classmethod
    def _read_from_xtc(cls, file_handler) -> np.ndarray:
        """
        Read data from an XTC file.

        Parameters
        ----------
        file_handler : md.formats.XTCTrajectoryFile
            The file handler for reading XTC files.

        Returns
        -------
        np.ndarray
            The data read from the XTC file.
        """
        return file_handler.read()

    @classmethod
    def _write_to_xtc(
        cls,
        file_handler: md.formats.XTCTrajectoryFile,
        positions: np.ndarray,
        iteration: np.ndarray,
        box_vectors: Optional[np.ndarray] = None,
    ):
        """
        Write position data to an XTC file.

        Parameters
        ----------
        file_handler : md.formats.XTCTrajectoryFile
            The file handler for writing to XTC files.
        positions : np.ndarray
            The positions to be written.
        iteration : np.ndarray
            The iteration numbers corresponding to the positions.
        box_vectors : Optional[np.ndarray], optional
            Box vectors for each position frame.
        """
        file_handler.write(
            positions,
            time=iteration,
            box=box_vectors,
        )
