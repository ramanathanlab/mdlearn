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

```
python3 -m venv env
source env/bin/activate
pip install -r requirements_dev.txt
pip install -e '.[torch]'
```

This mode will symlink the Python files from the current local source tree into the Python install.
Hence, if you modify a Python file, you do not need to reinstall mdlearn again and again.

4. Then, install pre-commit hooks: this will auto-format and auto-lint on commit to enforce consistent code style:
```
pre-commit install
pre-commit autoupdate
``` 

5. Ensure that you have a working mdlearn installation by running:

```
python -c "import mdlearn; print(mdlearn.__version__)"
```

## Unit Testing

We are planning to add a test suite in a future release which uses pytest for unit testing.

## Building Documentation

To build the documentation:

1. [Build and install](#developing-mdlearn) mdlearn from source.
2. The `requirements_dev.txt` contains all the dependencies needed to build the documentation.
3. Generate the documentation file via:
```
cd mdlearn/docs
make html
```
The docs are located in `mldearn/docs/build/html/index.html`.

To view the docs run: `open mldearn/docs/build/html/index.html`.

## Releasing to PyPI

To release a new version of mdlearn to PyPI:

1. Merge the `develop` branch into the `main` branch with an updated version number in [`mdlearn.__init__`](https://github.com/ramanathanlab/mdlearn/blob/main/mdlearn/__init__.py).
2. Make a new release on GitHub with the tag and name equal to the version number.
3. [Build and install](#developing-mdlearn) mdlearn from source.
4. Run the following commands:
```
python setup.py sdist
twine upload dist/*
```
