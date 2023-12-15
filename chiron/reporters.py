# This module implements classes that report simulation data.
from loguru import logger as log

import h5py
import numpy as np

from openmm.app import Topology


class SimulationReporter:
    def __init__(self, filename: str, topology: Topology, buffer_size: int = 1):
        """
        Initialize the SimulationReporter.

        Parameters
        ----------
        filename : str
            Name of the HDF5 file to write the simulation data.
        topology: openmm.Topology
        buffer_size : int, optional
            Number of data points to buffer before writing to disk (default is 1).

        """
        import mdtraj as md

        self.filename = filename
        self.buffer_size = buffer_size
        self.topology = topology
        self.buffer = {}
        self.h5file = h5py.File(filename, "a")
        log.info(f"Writing simulation data to {filename}")

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
                log.debug(f"Writing {key} to disk")
                log.debug(f"Buffer: {self.buffer[key]}")
                self._write_to_disk(key)

    def _write_to_disk(self, key):
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
            log.debug(f"Creating {key} in {self.filename}")
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
        return np.array(self.h5file[name])

    def get_mdtraj_trajectory(self):
        import mdtraj as md

        return md.Trajectory(
            xyz=self.get_property("traj"),
            topology=md.Topology.from_openmm(self.topology),
            # unitcell_lengths=self.get_property("box_vectors"),
            # unitcell_angles=self.get_property("box_angles"),
        )
