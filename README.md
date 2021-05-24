# mdlearn

mdlearn: Machine learning for molecular dynamics

[![Documentation Status](https://readthedocs.org/projects/mdlearn/badge/?version=latest)](https://mdlearn.readthedocs.io/en/latest/?badge=latest)

Details can be found in the [ducumentation](https://mdlearn.readthedocs.io/en/latest/).

## How to run

### Setup

Install `mdlearn` into a virtualenv with:

*Note*: For latest rapidsai install, see:  https://rapids.ai/start.html#get-rapids
```
conda create -p conda-env -c rapidsai -c nvidia -c conda-forge cuml=0.19 python=3.7 cudatoolkit=11.2
conda activate conda-env
export IBM_POWERAI_LICENSE_ACCEPT=yes
pip install -U scikit-learn
pip install -r requirements_dev.txt
pip install -e .
```

Then, install pre-commit hooks: this will auto-format and auto-lint _on commit_ to enforce consistent code style:

```
pre-commit install
pre-commit autoupdate
```

