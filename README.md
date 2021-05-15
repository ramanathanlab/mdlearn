# mdlearn

mdlearn: Machine learning for molecular dynamics

[![Documentation Status](https://readthedocs.org/projects/mdlearn/badge/?version=latest)](https://mdlearn.readthedocs.io/en/latest/?badge=latest)

Details can be found in the [ducumentation](https://mdlearn.readthedocs.io/en/latest/).

## How to run

### Setup

Install `mdlearn` into a virtualenv with:

```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements_dev.txt
pip install -e .
```

Then, install pre-commit hooks: this will auto-format and auto-lint _on commit_ to enforce consistent code style:

```
pre-commit install
pre-commit autoupdate
```

