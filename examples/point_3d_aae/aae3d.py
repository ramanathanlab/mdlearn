from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mdlearn.nn.models.aae.point_3d_aae import AAE3dTrainer

if __name__ == '__main__':
    """Train a 3D point autoencoder.

    You can customize this script further by specifying more AAE3dTrainer options.

    Example
    -------
    To run training:
    python aae3d.py \
        --input_path /nfs/lambda_stor_01/homes/abrace/projects/ddwe/data/CA_to_Alex/xyz_coords_aligned.npy \
        --scalars_path /nfs/lambda_stor_01/homes/abrace/projects/ddwe/data/rmsd.npy \
        --scalar_name rmsd \
        --output_path runs/run-0 \
        --batch_size 32 \
        --epochs 100 \
        --latent_dim 3 \
        --verbose

    To run evaluation only:
    python aae3d.py \
        --input_path /nfs/lambda_stor_01/homes/abrace/projects/ddwe/data/CA_to_Alex/xyz_coords_aligned.npy \
        --scalars_path /nfs/lambda_stor_01/homes/abrace/projects/ddwe/data/rmsd.npy \
        --scalar_name rmsd \
        --checkpoint_path runs/run-0/checkpoints/checkpoint-epoch-90.pt \
        --output_path runs/run-0 \
        --verbose \
        --eval_only
    """

    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path',
        type=str,
        help='The path to the input coordinates, an npy file with shape (N, num_points, 3)',
    )
    parser.add_argument(
        '--scalars_path',
        type=str,
        help='The path to the input scalars, an npy file with shape (N, 1)',
    )
    parser.add_argument(
        '--scalar_name',
        type=str,
        help='Name of the scalar to use for evaluation',
    )
    parser.add_argument(
        '--output_path',
        type=Path,
        help='The path to save the trained model',
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=None,
        help='The path to load a checkpoint model',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='The batch size for training',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='The number of epochs to train for',
    )
    parser.add_argument(
        '--latent_dim',
        type=int,
        default=3,
        help='The dimension of the latent space',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='The device to train on [cuda, cpu]',
    )
    parser.add_argument(
        '--plot_method',
        type=str,
        default='raw',
        help='The method to use for plotting the latent space [raw, PCA, TSNE]',
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Verbose output',
    )
    parser.add_argument(
        '--eval_only',
        action='store_true',
        help='Run evaluation only',
    )
    args = parser.parse_args()

    # Load the training data
    coords = np.load(args.input_path)
    # Reshape the data to (N_examples, 3, N_residues)
    coords = coords.transpose([0, 2, 1])
    # Print the training data shape
    print(
        f"{'Evaluate' if args.eval_only else 'Train'} on data with shape:",
        coords.shape,
    )

    # Load data for plotting evaluations
    scalar = np.load(args.scalars_path)
    scalars = {args.scalar_name: scalar}

    # Initialize the trainer
    trainer = AAE3dTrainer(
        num_points=coords.shape[2],  # number of residues
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        plot_method=args.plot_method,
        verbose=args.verbose,
    )

    if not args.eval_only:
        trainer.fit(
            X=coords,
            scalars=scalars,
            output_path=args.output_path,
            checkpoint=args.checkpoint_path,
        )

    if args.eval_only:
        # Checkpoint path must be provided for evaluation only mode
        if args.checkpoint_path is None:
            raise ValueError(
                'Must provide a checkpoint path for evaluation only mode',
            )

        # Evaluate the model
        z, _ = trainer.predict(X=coords, checkpoint=args.checkpoint_path)

        # Plot the latent space
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=scalar, cmap='viridis')

        # Add color bar for reference and label it
        cbar = plt.colorbar(sc)
        cbar.set_label(args.scalar_name)

        # Set labels
        ax.set_xlabel(r'$z_0$')
        ax.set_ylabel(r'$z_1$')
        ax.set_zlabel(r'$z_2$')

        # Save the plot as a high-quality PNG file
        png_file = args.output_path / f'{args.scalar_name}_latent_space.png'
        plt.savefig(png_file, dpi=300, bbox_inches='tight')
