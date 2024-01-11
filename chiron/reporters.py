# This module implements classes that report simulation data.
from loguru import logger as log

import h5py
import numpy as np

from openmm.app import Topology


class BaseReporter:
    _directory = None

    @classmethod
    def set_directory(cls, directory: str):
        cls._directory = directory

    @classmethod
    def get_directory(cls):
        from pathlib import Path

        if cls._directory is None:
            log.debug(
                f"No directory set, using current working directory: {Path.cwd()}"
            )
            return Path.cwd()
        return Path(cls._directory)


import pathlib


class _SimulationReporter:
    def __init__(self, file_path: pathlib.Path, buffer_size: int = 10):
        """
        Initialize the SimulationReporter.

        Parameters
        ----------
        filename : str
            Name of the HDF5 file to write the simulation data.
        """
        if file_path.suffix != ".h5":
            file_path = file_path.with_suffix(".h5")
        self.file_path = file_path
        log.info(f"Writing simulation data to {self.file_path}")

        self.buffer_size = buffer_size
        self.buffer = {}
        self.h5file = h5py.File(self.file_path, "a")

    def get_available_keys(self):
        return self.h5file.keys()

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
                self.buffer[key] = []
            self.buffer[key].append(value)

            if len(self.buffer[key]) >= self.buffer_size:
                self._write_to_disk(key)

    def _write_to_disk(self, key: str):
        """
        Write buffered data of a given key to the HDF5 file.

        Parameters
        ----------
        key : str
            The key of the data to write to disk.

        """
        data = np.array(self.buffer[key])
        if key in self.h5file:
            dset = self.h5file[key]
            dset.resize((dset.shape[0] + data.shape[0],) + data.shape[1:])
            dset[-data.shape[0] :] = data
        else:
            log.debug(f"Creating {key} in {self.file_path}")
            self.h5file.create_dataset(
                key, data=data, maxshape=(None,) + data.shape[1:], chunks=True
            )

        self.buffer[key] = []

    def close(self):
        """
        Write any remaining data in the buffer to disk and close the HDF5 file.

        """
        for key in self.buffer:
            if self.buffer[key]:
                self._write_to_disk(key)
        self.h5file.close()

    def get_property(self, name: str):
        """
        Get the property from the HDF5 file.

        Parameters
        ----------
        name : str
            Name of the property to get.

        Returns
        -------
        np.ndarray
            The property.

        """
        if name not in self.h5file:
            log.warning(f"{name} not in HDF5 file")
            return None
        else:
            return np.array(self.h5file[name])


from typing import Optional


class MCReporter(_SimulationReporter):
    _name = "mc_reporter"

    def __init__(self, name: str, buffer_size: int = 1):
        filename = LangevinDynamicsReporter.get_name()
        directory = BaseReporter.get_directory()
        import os

        os.makedirs(directory, exist_ok=True)
        self.file_path = directory / f"{filename}_{name}"

        super().__init__(file_path=self.file_path, buffer_size=buffer_size)

    @classmethod
    def get_name(cls):
        return cls._name


class LangevinDynamicsReporter(_SimulationReporter):
    _name = "langevin_reporter"

    def __init__(
        self, name: str, buffer_size: int = 1, topology: Optional[Topology] = None
    ):
        """
        Initialize the SimulationReporter.

        Parameters
        ----------
        name : str
            Name of the HDF5 file to write the simulation data.
        buffer_size : int, optional
            Number of data points to buffer before writing to disk (default is 1).
        topology: openmm.Topology, optional
            Topology of the system to generate the mdtraj trajectory.
        """
        filename = LangevinDynamicsReporter.get_name()
        directory = BaseReporter.get_directory()
        import os

        os.makedirs(directory, exist_ok=True)
        self.file_path = directory / f"{filename}_{name}"

        self.topology = topology
        super().__init__(file_path=self.file_path, buffer_size=buffer_size)

    @classmethod
    def get_name(cls):
        return cls._name

    def get_mdtraj_trajectory(self):
        import mdtraj as md

        return md.Trajectory(
            xyz=self.get_property("traj"),
            topology=md.Topology.from_openmm(self.topology),
            unitcell_lengths=self.get_property("box_vectors"),
            unitcell_angles=self.get_property("box_angles"),
        )


class MultistateReporter:
    def __init__(self, path_to_dir: str) -> None:
        self.path_to_dir = path_to_dir

    def _write_trajectories():
        pass

    def _write_energies():
        pass

    def _write_states():
        pass
