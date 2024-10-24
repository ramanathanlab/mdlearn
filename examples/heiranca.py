from __future__ import annotations

import itertools
from pathlib import Path
from typing import TYPE_CHECKING

import MDAnalysis as mda
import numpy as np

from mdlearn.data.preprocess.align import iterative_means_align
from mdlearn.data.preprocess.decorrelation.spatial import SD2
from mdlearn.data.preprocess.decorrelation.spatial import SD4
from mdlearn.nn.models.ae.linear import LinearAETrainer
from mdlearn.utils import PathLike

if TYPE_CHECKING:
    import numpy.typing as npt


def parse_position_groups(
    pdb_file: PathLike,
    traj_file: PathLike,
    selections: list[str],
) -> dict[str, npt.ArrayLike]:
    u = mda.Universe(str(pdb_file), str(traj_file))
    atom_groups = {sel: u.select_atoms(sel) for sel in selections}
    position_groups = {sel: [] for sel in selections}
    for _ in u.trajectory:
        for sel in selections:
            positions = atom_groups[sel].transpose()
            position_groups[sel].append(positions)
    # Convert from list to np.array
    position_groups = {k: np.array(v) for k, v in position_groups.items()}
    return position_groups


if __name__ == '__main__':
    pdb_file = ''
    traj_file = ''
    selections = ['protein and name CA']
    alignment_workers = 10
    num_sd4_subspaces = 60
    latent_dim = 8
    autoencoder_params = dict(
        input_dim=num_sd4_subspaces,
        latent_dim=latent_dim,
        hidden_neurons=[32, 16],
        epochs=500,
        checkpoint_log_every=500,
        plot_method=None,
        device='cpu',
    )
    output_path = Path('run-0')
    # Autoencoder outputs for each selection
    selection_run_path = output_path / 'selection_runs'
    # Autoencoder output for the aggregate selection
    aggregate_run_path = output_path / 'aggregate_run'
    output_path.mkdir()
    selection_run_path.mkdir()
    aggregate_run_path.mkdir()

    # Parse atom group positions from trajectory file
    position_groups = parse_position_groups(pdb_file, traj_file, selections)
    all_positions = np.concatenate(list(position_groups.values()), axis=2)

    # Run alignment of all positions and save results
    _, _, e_rmsd, aligned_coords = iterative_means_align(
        all_positions,
        num_workers=alignment_workers,
    )
    np.save(output_path / 'e_rmsd.npy', e_rmsd)
    np.save(output_path / 'all_positions.npy', all_positions)
    np.save(output_path / 'aligned_coords.npy', aligned_coords)

    group_lengths = [
        positions.shape[2] for positions in position_groups.values()
    ]
    start_ind = 0
    aligned_position_groups = {}
    for sel, length in zip(selections, group_lengths):
        aligned_position_groups[sel] = aligned_coords[
            :,
            :,
            start_ind : start_ind + length,
        ]
        start_ind += length
    # We now have a dictionary with aligned positions for each selection
    # Dict[selection: array of shape (N, 3, num_atoms_in_selection)]
    # where N is the number of frames

    # Compute and store the latent space (num_frames, latent_dim) for each selection
    latent_spaces = {}
    for i, (selection, positions) in enumerate(
        aligned_position_groups.items(),
    ):
        selection_path = selection_run_path / f'selection-{i}'
        # Log the selection string to document the model directories
        with open(selection_path / 'selection.txt') as f:
            f.write(selection)

        num_frames, _, num_atoms = positions.shape

        # Run SD2 and save results
        Y, S, B, U = SD2(
            positions.reshape(num_frames, num_atoms * 3),
            m=num_atoms * 3,
        )
        np.save(selection_path / 'Y.npy', Y)
        np.save(selection_path / 'S.npy', S)
        np.save(selection_path / 'B.npy', B)
        np.save(selection_path / 'U.npy', U)

        # Run SD4 and save results
        W = SD4(
            Y[0:num_sd4_subspaces, :],
            m=num_sd4_subspaces,
            U=U[0:num_sd4_subspaces, :],
        )
        coords = np.reshape(positions, (num_frames, 3 * num_atoms)).T
        avg_coords = np.mean(coords, 1)
        tmp = np.reshape(
            np.tile(avg_coords, num_frames),
            (num_frames, 3 * num_atoms),
        ).T
        devs = coords - tmp
        sd4_projection: npt.ArrayLike = W.dot(devs).T
        np.save(selection_path / 'W.npy', W)
        np.save(selection_path / 'sd4_projection.npy', sd4_projection)
        # At this point, the SD4 projection results in an array with
        # shape (num_frames, num_sd4_subspaces)

        # Run autoencoder and save results
        autoencoder = LinearAETrainer(**autoencoder_params)
        autoencoder.fit(
            sd4_projection,
            output_path=selection_path / 'autoencoder',
        )
        z, _ = autoencoder.predict(sd4_projection)
        latent_spaces[selection] = z
        np.save(selection_path / 'z.npy', z)

    # # Combine each of the (num_frames, latent_dim) latent spaces
    # # into a single tensor of shape (num_frames, num_selections, latent_dim)
    # combined_z = np.zeros((num_frames, len(selections), latent_dim))
    # for i, z in enumerate(latent_spaces):
    #     combined_z[:, i, :] = z

    # Combine each of the (num_frames, latent_dim) latent spaces
    # into a single tensor of shape (num_frames, len(selections) * latent_dim)
    # This approach also works for heterogeneous latent space sizes
    # (just need to generalize the latent_dim size variable)
    combined_z = np.zeros((num_frames, len(selections) * latent_dim))
    start_ind = 0
    for i, z in enumerate(latent_spaces):
        combined_z[:, start_ind : start_ind + latent_dim] = z
        start_ind += latent_dim

    # Learn a representation of the entire system, given the combined
    # latent spaces of the individual selections

    # Find powers of 2, smaller than combined_z.shape[1] and larger than latent_dim
    hidden_neurons = []
    for i in itertools.count():
        power = 2**i
        if power <= latent_dim:
            pass
        elif power < combined_z.shape[1]:
            hidden_neurons.append(power)
        else:
            break
    if not hidden_neurons:
        hidden_neurons = [(combined_z.shape[1] + latent_dim) // 2]
    hidden_neurons = list(reversed(hidden_neurons))

    autoencoder = LinearAETrainer(
        input_dim=combined_z.shape[1],
        latent_dim=latent_dim,
        hidden_neurons=hidden_neurons,
        epochs=500,
        checkpoint_log_every=500,
        plot_method=None,
        device='cpu',
    )
    autoencoder.fit(combined_z, output_path=aggregate_run_path)
    z, _ = autoencoder.predict(combined_z)
