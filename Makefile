.DEFAULT_GOAL := all
isort = isort mdlearn examples test
black = black --target-version py37 mdlearn examples test

.PHONY: format
format:
	$(isort)
	$(black)

.PHONY: lint
lint:
	$(black) --check --diff
	flake8 mdlearn/ examples/ test/
	#pylint mdlearn/ #examples/ test/
	#pydocstyle mdlearn/


.PHONY: mypy
mypy:
	mypy --config-file setup.cfg --package mdlearn
	mypy --config-file setup.cfg mdlearn/
	mypy --config-file setup.cfg examples/

.PHONY: all
all: format lint mypy