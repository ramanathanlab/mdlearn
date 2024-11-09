"""Module for extracting outputs from molecular dynamics trajectories."""

from __future__ import annotations

import functools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any
from typing import Protocol

import MDAnalysis
import numpy as np
from MDAnalysis.analysis import distances
from MDAnalysis.analysis import rms
from MDAnalysis.analysis.align import AlignTraj
from tqdm import tqdm


class SimulationPreprocessor(Protocol):
    """Protocol for simulation data preprocessors."""

    def __init__(
        self,
        top_file: Path | str,
        traj_file: Path | str,
        *args: Any,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the simulation preprocessor.

        Parameters
        ----------
        top_file : Path | str
            Topology file of the simulation.
        traj_file : Path | str
            Trajectory file of the simulation.
        *args : Any
            Positional arguments for the preprocessor.
        **kwargs : dict[str, Any]
            Keyword arguments for the preprocessor.
        """
        ...

    def get(self) -> np.ndarray:
        """Get simulation data from a trajectory file.

        Returns
        -------
        np.ndarray
            Simulation data from the trajectory.
        """
        ...


class CoordinatePreprocessor:
    """Process coordinates from a MD trajectory."""

    def __init__(
        self,
        top_file: Path | str,
        traj_file: Path | str,
        ref_file: Path | str,
        selection: str = 'protein and name CA',
    ) -> None:
        """Initialize the coordinate preprocessor.

        Parameters
        ----------
        top_file : Path | str
            Topology file of the simulation.
        traj_file : Path | str
            Trajectory file of the simulation.
        ref_file : Path | str
            Reference structure file to align the trajectory.
        selection : str
            Atom selection string for the reference structure,
            defaults to 'protein and name CA'.
        """
        # Load simulation and reference structures
        self.sim = MDAnalysis.Universe(str(top_file), str(traj_file))
        ref = MDAnalysis.Universe(str(ref_file))

        mobile_atoms = self.sim.select_atoms('protein and name CA')
        ref_atoms = ref.select_atoms('protein and name CA')
        print(mobile_atoms)  # Check if any atoms are selected
        print(ref_atoms)  # Ensure ref atoms are selected too

        # Align trajectory to a reference structure
        AlignTraj(self.sim, ref, select=selection, in_memory=True).run()

        # Atom selection for reference
        self.atoms = self.sim.select_atoms(selection)

    def get(self) -> np.ndarray:
        """Get coordinates of a trajectory file.

        Returns
        -------
        np.ndarray
            Coordinates of the trajectory. The shape of the array is
            (n_frames, n_atoms, 3), where n_frames is the number of
            frames in the trajectory, n_atoms is the number of atoms
            in the selection, and 3 corresponds to x, y, and z.
        """
        return np.array(
            [self.atoms.positions for _ in self.sim.trajectory],
        )


class ContactMapPreprocessor:
    """Process contact maps from a MD trajectory."""

    def __init__(
        self,
        top_file: Path | str,
        traj_file: Path | str,
        cutoff: float = 8.0,
        selection: str = 'protein and name CA',
    ) -> None:
        """Initialize the contact map preprocessor.

        Parameters
        ----------
        top_file : Path | str
            Topology file of the simulation.
        traj_file : Path | str
            Trajectory file of the simulation.
        cutoff : float
            Cutoff distance (in Angstroms) for contact map calculation,
            defaults to 8.0.
        selection : str
            Atom selection string for the reference structure,
            defaults to 'protein and name CA'.
        """
        # Load simulation and reference structures and select atoms
        self.sim = MDAnalysis.Universe(str(top_file), str(traj_file))
        self.atoms = self.sim.select_atoms(selection)

        # Partial function for contact map calculation
        self.contact_map_fn = functools.partial(
            distances.contact_matrix,
            cutoff=cutoff,
            returntype='sparse',
            box=self.sim.atoms.dimensions,
        )

    def _get_contact_maps(self, positions: np.ndarray) -> np.ndarray:
        # Compute contact map of current frame (scipy lil_matrix form)
        cm = self.contact_map_fn(positions)

        # Convert to COO-sparse format
        coo_cm = cm.tocoo()
        row = coo_cm.row.astype('int16')
        col = coo_cm.col.astype('int16')

        # Concatenate the row and column indices of the ones in the contact map
        return np.concatenate([row, col])

    def get(self) -> np.ndarray:
        """Get contact maps of a trajectory file.

        Returns
        -------
        np.ndarray
            The contact maps of the trajectory with shape (n_frames, *)
            where * is a ragged dimension containing the concatenated
            row and column indices of the ones in the contact map.
        """
        # Compute contact maps for each frame
        contact_maps = [
            self._get_contact_maps(self.atoms.positions)
            for _ in self.sim.trajectory
        ]
        # Convert to object array to handle ragged dimensions
        return np.array(contact_maps, dtype=object)


class RmsdPreprocessor:
    """Process RMSD from a MD trajectory."""

    def __init__(
        self,
        top_file: Path | str,
        traj_file: Path | str,
        ref_file: Path | str,
        selection: str = 'protein and name CA',
    ) -> None:
        """Initialize the RMSD preprocessor.

        Parameters
        ----------
        top_file : Path | str
            Topology file of the simulation.
        traj_file : Path | str
            Trajectory file of the simulation.
        ref_file : Path | str
            Reference structure file to calculate RMSD.
        selection : str
            Atom selection string for the reference structure,
            defaults to 'protein and name CA'.
        """
        # Load simulation and reference structures and select atoms
        self.sim = MDAnalysis.Universe(str(top_file), str(traj_file))
        self.atoms = self.sim.select_atoms(selection)

        # Get atomic coordinates of reference atoms
        ref = MDAnalysis.Universe(str(ref_file))
        ref_positions = ref.select_atoms(selection).positions.copy()

        # Partial function for RMSD calculation
        self.rmsd_fn = functools.partial(
            rms.rmsd,
            b=ref_positions,
            center=True,
            superposition=True,
        )

    def get(self) -> np.ndarray:
        """Get RMSD to reference state of a trajectory file.

        Returns
        -------
        np.ndarray
            RMSD to reference state of the trajectory. The shape of the
            array is (n_frames,), where n_frames is the number of frames
            in the trajectory.
        """
        return np.array(
            [self.rmsd_fn(self.atoms.positions) for _ in self.sim.trajectory],
        )


PREPROCESSORS = {
    'coordinates': CoordinatePreprocessor,
    'contact_maps': ContactMapPreprocessor,
    'rmsd': RmsdPreprocessor,
}


def preprocess(
    top_file: Path | str,
    traj_file: Path | str,
    output_dir: Path | str,
    topic: str,
    **kwargs: Any,
) -> None:
    """Preprocess simulation data from a trajectory file.

    Parameters
    ----------
    top_file : Path | str
        Topology file of the simulation.
    traj_file : Path | str
        Trajectory file of the simulation.
    output_dir : Path | str
        Output directory to save the preprocessed data.
    topic : str
        Topic of the simulation data.
    **kwargs : Any
        Keyword arguments for the preprocessor.
    """
    # Check if the topic is valid
    if topic not in PREPROCESSORS:
        raise ValueError(f'Invalid topic: {topic}')

    # Get preprocessor for the specified topic
    preprocessor = PREPROCESSORS[topic](traj_file, top_file, **kwargs)

    # Preprocess the simulation data
    data = preprocessor.get()

    # Create the output directory if it does not exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save the preprocessed data to disk
    np.save(Path(output_dir) / f'{topic}.npy', data)


def parallel_preprocess(
    topic: str,
    input_dir: Path | str,
    output_dir: Path | str,
    top_ext: str = '.pdb',
    traj_ext: str = '.dcd',
    num_workers: int = 10,
    **kwargs: Any,
) -> None:
    """Preprocess simulation data from many trajectory files in parallel.

    Parameters
    ----------
    topic : str
        Topic/name of the preprocessor.
    input_dir : Path | str
        Input directory containing the trajectory files.
    output_dir : Path | str
        Output directory to save the preprocessed data.
    top_ext : str
        Extension of the topology files, defaults to '.pdb'.
    traj_ext : str
        Extension of the trajectory files, defaults to '.dcd'.
    num_workers : int
        Number of workers for parallel processing, defaults to 10.
    **kwargs : Any
        Keyword arguments for the preprocessor.
    """
    # Collect the traj files from the nested subdirectories
    traj_files = list(Path(input_dir).rglob(f'*{traj_ext}'))

    # Collect the topology files from the nested subdirectories
    top_files = [next(f.parent.glob(f'*{top_ext}')) for f in traj_files]

    # Create output directories for each trajectory file
    output_dirs = [Path(output_dir) / f.parent.name for f in traj_files]

    # Create a partial function for the worker with the fixed keyword arguments
    worker_fn = functools.partial(
        preprocess,
        topic=topic,
        **kwargs,
    )

    # Zip the arguments for the worker function
    args = list(zip(top_files, traj_files, output_dirs))

    # Process the trajectory files in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for _ in tqdm(executor.map(worker_fn, *zip(*args)), total=len(args)):
            pass
