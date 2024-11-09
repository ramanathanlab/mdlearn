"""CLI for the mdlearn package."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer()

# Define a subcommand for preprocessing
preprocess_app = typer.Typer()
app.add_typer(preprocess_app, name='preprocess')


# Define shared options
preprocess_options = {
    'input_dir': typer.Option(
        ...,
        '--input_dir',
        '-i',
        help='Input directory containing subdirectories with topology'
        ' and trajectory files.',
    ),
    'output_dir': typer.Option(
        ...,
        '--output_dir',
        '-o',
        help='Output directory to save the preprocessed data.',
    ),
    'top_ext': typer.Option(
        '.pdb',
        '--top_ext',
        '-t',
        help='Extension of the topology files, defaults to ".pdb".',
    ),
    'traj_ext': typer.Option(
        '.dcd',
        '--traj_ext',
        '-t',
        help='Extension of the trajectory files, defaults to ".dcd".',
    ),
    'num_workers': typer.Option(
        1,
        '--num_workers',
        '-n',
        help='Number of parallel workers to use for preprocessing, '
        'defaults to 1.',
    ),
    'selection': typer.Option(
        'protein and name CA',
        '--selection',
        '-s',
        help='Selection string to extract atoms from the trajectory, '
        'defaults to "protein and name CA".',
    ),
    'ref_file': typer.Option(
        ...,
        '--ref_file',
        '-r',
        help='Reference structure file to align the trajectory.',
    ),
}


@preprocess_app.command()
def coordinates(
    input_dir: Path = preprocess_options['input_dir'],
    output_dir: Path = preprocess_options['output_dir'],
    top_ext: str = preprocess_options['top_ext'],
    traj_ext: str = preprocess_options['traj_ext'],
    num_workers: int = preprocess_options['num_workers'],
    selection: str = preprocess_options['selection'],
    ref_file: Path = preprocess_options['ref_file'],
) -> None:
    """Preprocess coordinates from a MD trajectory."""
    from mdlearn.data.preprocess.simulation import parallel_preprocess

    # Preprocess the simulation data
    parallel_preprocess(
        topic='coordinates',
        input_dir=input_dir,
        output_dir=output_dir,
        top_ext=top_ext,
        traj_ext=traj_ext,
        num_workers=num_workers,
        selection=selection,
        ref_file=ref_file,
    )


@preprocess_app.command()
def contact_map(
    input_dir: Path = preprocess_options['input_dir'],
    output_dir: Path = preprocess_options['output_dir'],
    top_ext: str = preprocess_options['top_ext'],
    traj_ext: str = preprocess_options['traj_ext'],
    num_workers: int = preprocess_options['num_workers'],
    selection: str = preprocess_options['selection'],
    cutoff: float = typer.Option(
        8.0,
        '--cutoff',
        '-c',
        help='Cutoff distance (in Angstroms) for contact map calculation, '
        'defaults to 8.0.',
    ),
) -> None:
    """Preprocess contact maps from a MD trajectory."""
    from mdlearn.data.preprocess.simulation import parallel_preprocess

    # Preprocess the simulation data
    parallel_preprocess(
        topic='contact_map',
        input_dir=input_dir,
        output_dir=output_dir,
        top_ext=top_ext,
        traj_ext=traj_ext,
        num_workers=num_workers,
        selection=selection,
        cutoff=cutoff,
    )


@preprocess_app.command()
def rmsd(
    input_dir: Path = preprocess_options['input_dir'],
    output_dir: Path = preprocess_options['output_dir'],
    top_ext: str = preprocess_options['top_ext'],
    traj_ext: str = preprocess_options['traj_ext'],
    num_workers: int = preprocess_options['num_workers'],
    selection: str = preprocess_options['selection'],
    ref_file: Path = preprocess_options['ref_file'],
) -> None:
    """Preprocess RMSD from a MD trajectory."""
    from mdlearn.data.preprocess.simulation import parallel_preprocess

    # Preprocess the simulation data
    parallel_preprocess(
        topic='rmsd',
        input_dir=input_dir,
        output_dir=output_dir,
        top_ext=top_ext,
        traj_ext=traj_ext,
        num_workers=num_workers,
        selection=selection,
        ref_file=ref_file,
    )


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == '__main__':
    main()
