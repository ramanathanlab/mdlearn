# Contributing to mdlearn

If you are interested in contributing to mdearn, your contributions will fall into two categories:

1. You want to implement a new feature:
    - In general, we accept any features as long as they fit the scope of this package. If you are unsure about this or need help on the design/implementation of your feature, post about it in an issue.
2. You want to fix a bug:
    - Please post an issue using the Bug template which provides a clear and concise description of what the bug was.

Once you finish implementing a feature or bug-fix, please send a Pull Request to https://github.com/ramanathanlab/mdlearn.

## Developing mdlearn

To develop mdlearn on your machine, please follow these instructions:


1. Clone a copy of mdlearn from source:

```
git clone https://github.com/ramanathanlab/mdlearn.git
cd mdlearn
```

2. If you already have a mdlearn from source, update it:

```
git pull
```

3. Install mdlearn in `develop` mode:

For development, it is recommended to use a virtual environment. The following
commands will create a virtual environment, install the package in editable
mode, and install the pre-commit hooks.
```bash
python -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -e '.[dev,docs,torch]'
pre-commit install
```

## Unit Testing

To run the test suite:

First, [Build and install](#developing-mdlearn) mdlearn from source.

To test the code, run the following command:
```bash
pre-commit run --all-files
tox -e py312
```

If contributing, please add a `test_<module_name>.py` in the `test/` directory
in a subdirectory that matches the mdlearn package directory structure. Inside,
`test_<module_name>.py` implement test functions using pytest.

## Building Documentation

To build the documentation:

1. [Build and install](#developing-mdlearn) mdlearn from source.
2. Generate the documentation file via:
```
cd docs
make html
```
The docs are located in `build/html/index.html`.

To view the docs run: `open build/html/index.html`.

## Releasing to PyPI

To release a new version of mdlearn to PyPI:

1. Merge the `develop` branch into the `main` branch with an updated version number in `pyproject.toml`.
2. Make a new release on GitHub with the tag and name equal to the version number.
3. [Build and install](#developing-mdlearn) mdlearn from source.
4. Run the following commands:
```
python3 -m build
twine upload dist/*
```
