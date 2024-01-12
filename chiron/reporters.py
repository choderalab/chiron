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
        self.workdir = file_path.parent
        log.info(f"Writing simulation data to {self.file_path}")

        self.buffer_size = buffer_size
        self.buffer = {}
        self.h5file = h5py.File(self.file_path, "a")

    @property
    def properties_to_report(self):
        return self._properties_to_report

    @properties_to_report.setter
    def properties_to_report(self, properties: List[str]):
        self._properties_to_report = properties

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
                if key == "positions" and hasattr(self, "_write_to_trajectory"):
                    log.debug(f"Writing positions to trajectory")
                    log.debug(f"Positions: {value['xyz']}")
                    self._write_to_trajectory(
                        positions=value["xyz"],
                        replica_id=value["replica_id"],
                    )
                else:
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

    def reset_reporter_file(self):
        # delete the reporter files
        import os

        # if file exists, delete it
        if os.path.exists(self.file_path):
            log.debug(f"Deleting {self.file_path}")
            os.remove(self.file_path)
            self.h5file = h5py.File(self.file_path, "a")

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


class MultistateReporter(_SimulationReporter):
    _name = "multistate_reporter"
    _default_properties = [
        "positions",
        "box_vectors",
        "u_kn",
        "state_index",
        "time",
    ]

    def __init__(self, buffer_size: int = 1) -> None:
        filename = MultistateReporter.get_name()
        directory = BaseReporter.get_directory()
        import os

        os.makedirs(directory, exist_ok=True)
        self.file_path = directory / f"{filename}.h5"

        super().__init__(file_path=self.file_path, buffer_size=buffer_size)
        self._properties_to_report = MultistateReporter._default_properties
        self._file_handle = {}

    @classmethod
    def get_name(cls):
        return cls._name

    def _write_to_trajectory(self, positions: np.ndarray, replica_id: int):
        import mdtraj as md

        # append to xtc trajectory the new positions
        file_name = f"replica_{replica_id}"
        if self._file_handle.get(file_name) is None:
            self._file_handle[file_name] = md.formats.XTCTrajectoryFile(
                f"{self.workdir}/{file_name}.xtc", mode="w"
            )

        open_xtc_file = self._file_handle[file_name]

        open_xtc_file.write(
            positions,
            #            time=iteration,
            box=self.get_property("box_vectors"),
        )


class MCReporter(_SimulationReporter):
    _name = "mc_reporter"

    def __init__(self, buffer_size: int = 1) -> None:
        filename = MCReporter.get_name()
        directory = BaseReporter.get_directory()
        import os

        os.makedirs(directory, exist_ok=True)
        self.file_path = directory / f"{filename}.h5"

        super().__init__(file_path=self.file_path, buffer_size=buffer_size)

    @classmethod
    def get_name(cls):
        return cls._name


class LangevinDynamicsReporter(_SimulationReporter):
    _name = "langevin_reporter"
    _default_properties = ["trajectory", "box_vectors", "potential_energy", "time"]

    def __init__(
        self,
        buffer_size: int = 1,
        topology: Optional[Topology] = None,
    ):
        """
        Initialize the SimulationReporter.

        Parameters
        ----------
        name_suffix : str
            Prefix of the HDF5 file to write the simulation data.
        buffer_size : int, optional
            Number of data points to buffer before writing to disk (default is 1).
        topology: openmm.Topology, optional
            Topology of the system to generate the mdtraj trajectory.
        """
        filename = LangevinDynamicsReporter.get_name()
        directory = BaseReporter.get_directory()
        import os

        os.makedirs(directory, exist_ok=True)
        self.file_path = directory / f"{filename}.h5"

        self.topology = topology
        super().__init__(file_path=self.file_path, buffer_size=buffer_size)
        self._default_properties = LangevinDynamicsReporter._default_properties

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
