# mdlearn

mdlearn: Machine learning for molecular dynamics

[![Documentation Status](https://readthedocs.org/projects/mdlearn/badge/?version=latest)](https://mdlearn.readthedocs.io/en/latest/?badge=latest)

Details can be found in the [ducumentation](https://mdlearn.readthedocs.io/en/latest/).

## How to run

### Setup

Install `mdlearn` into a virtualenv with:

```
conda create -p conda-env
conda activate conda-env
conda config --add channels rapidsai
conda install --channel "rapidsai" cuml
export IBM_POWERAI_LICENSE_ACCEPT=yes
pip install --upgrade pip setuptools wheel
pip install -r requirements_dev.txt
pip install -e .
```

Then, install pre-commit hooks: this will auto-format and auto-lint _on commit_ to enforce consistent code style:

```
pre-commit install
pre-commit autoupdate
```

