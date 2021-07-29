# mdlearn

Machine learning for molecular dynamics.

[![Documentation Status](https://readthedocs.org/projects/mdlearn/badge/?version=latest)](https://mdlearn.readthedocs.io/en/latest/?badge=latest)

Details can be found in the [documentation](https://mdlearn.readthedocs.io/en/latest/).

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Contributing](#contributing)
4. [History](#history)
5. [Acknowledgments](#acknowledgments)
6. [License](#license)

## Installation

Install `mdlearn` into a conda-env with:

*Note*: For latest rapidsai install, see:  https://rapids.ai/start.html#get-rapids
```
conda create -p conda-env -c rapidsai -c nvidia -c conda-forge cuml=0.19 python=3.7 cudatoolkit=11.2
conda activate conda-env
export IBM_POWERAI_LICENSE_ACCEPT=yes
pip install -U scikit-learn
pip install -r requirements_dev.txt
pip install -e '.[torch]'
```

Then, install pre-commit hooks: this will auto-format and auto-lint _on commit_ to enforce consistent code style:

```
pre-commit install
pre-commit autoupdate
```

## Usage

TODO

## Contributing

Please report **bugs**, **enhancement requests**, or **questions** through the [Issue Tracker](https://github.com/ramanathanlab/mdlearn/issues).

If you are looking to contribute, please follow these steps:

1. Fork it!
2. Create your feature branch: `git checkout -b feature/my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/my-new-feature`
5. Submit a pull request :D

## History

TODO

## Acknowledgments

- We thank [Matthias Fey](https://github.com/rusty1s) from [*PyTorch Geometric*](https://github.com/rusty1s/pytorch_geometric) for inspiring the design of our neural network base classes and other PyTorch helper functions.

## License

mdlearn has a MIT license, as seen in the [LICENSE](https://github.com/ramanathanlab/mdlearn/blob/main/LICENSE) file.

