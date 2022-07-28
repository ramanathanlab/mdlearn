# mdlearn

[![PyPI version](https://badge.fury.io/py/mdlearn.svg)](https://badge.fury.io/py/mdlearn)
[![Documentation Status](https://readthedocs.org/projects/mdlearn/badge/?version=latest)](https://mdlearn.readthedocs.io/en/latest/?badge=latest)

mdlearn is a Python library for analyzing molecular dynamics with machine learning. It contains [PyTorch](https://pytorch.org/) implementations of several deep learning methods such as autoencoders, as well as preprocessing functions which include the [kabsch alignment](https://en.wikipedia.org/wiki/Kabsch_algorithm) algorithm and higher-order statistical methods like [quasi-anharmonic analysis](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0015827).

For more details and specific examples of how to use mdlearn, please see our [documentation](https://mdlearn.readthedocs.io/en/latest/).

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Contributing](#contributing)
4. [Acknowledgments](#acknowledgments)
5. [License](#license)

## Installation

### Install latest version with PyPI 

If you have access to an NVIDIA GPU, we highly recommend installing mdlearn into a Conda environment which contains [RAPIDS](https://rapids.ai/) to accelerate t-SNE computations useful for visualizing the model results during training. For the latest [RAPIDS](https://rapids.ai/) version, see [here](https://rapids.ai/start.html#get-rapids). If you don't have GPU support, mdlearn will still work on CPU by using the [scikit-learn](https://scikit-learn.org/stable/) implementation.

Run the following commands with updated versions to create a conda environment:
```
conda create -p conda-env -c rapidsai -c nvidia -c conda-forge cuml=0.19 python=3.7 cudatoolkit=11.2
conda activate conda-env
export IBM_POWERAI_LICENSE_ACCEPT=yes
pip install -U scikit-learn
```

Then install mdlearn via: `pip install mdlearn`. 

Some systems require [PyTorch](https://pytorch.org/) to be built from source instead of installed via PyPI or Conda, for this reason we made torch an optional dependency. However, it can be installed with mdlearn by running `pip install 'mdlearn[torch]'` for convenience. Installing this way will also install the `wandb` package. Please check that **torch version >= 1.7**. 

## Usage

Train an autoencoder model with only a few lines of code!

```python
from mdlearn.nn.models.ae.linear import LinearAETrainer

# Initialize autoencoder model
trainer = LinearAETrainer(
    input_dim=40, latent_dim=3, hidden_neurons=[32, 16, 8], epochs=100
)

# Train autoencoder on (N, 40) dimensional data
trainer.fit(X, output_path="./run")

# Generate latent embeddings in inference mode
z, loss = trainer.predict(X)
```

## Contributing

Please report **bugs**, **enhancement requests**, or **questions** through the [Issue Tracker](https://github.com/ramanathanlab/mdlearn/issues).

If you are looking to contribute, please see [`CONTRIBUTING.md`](https://github.com/ramanathanlab/mdlearn/blob/main/CONTRIBUTING.md).


## Acknowledgments

- We thank [Matthias Fey](https://github.com/rusty1s) from [*PyTorch Geometric*](https://github.com/rusty1s/pytorch_geometric) for inspiring the design of our neural network base classes and other [PyTorch](https://pytorch.org/) helper functions.

## License

mdlearn has a MIT license, as seen in the [LICENSE](https://github.com/ramanathanlab/mdlearn/blob/main/LICENSE) file.

