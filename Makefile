#* Variables
SHELL := /usr/bin/env bash
PYTHON := python
PYTHONPATH := `pwd`

#* Docker variables
IMAGE := roc
VERSION := latest

#* Poetry
.PHONY: poetry-download
poetry-download:
	curl -sSL https://install.python-poetry.org | python3 -

.PHONY: poetry-remove
poetry-remove:
	curl -sSL https://install.python-poetry.org | $(PYTHON) - --uninstall

#* Installation
.PHONY: setup
setup: install pre-commit-install

.PHONY: install
install:
	poetry config virtualenvs.in-project true
	poetry lock -n && poetry export --without-hashes > requirements.txt
	poetry install -n
	-poetry run mypy --install-types --non-interactive ./

.PHONY: pre-commit-install
pre-commit-install:
	poetry run pre-commit install --hook-type pre-commit --hook-type pre-push

#* Formatters
.PHONY: codestyle
codestyle:
	poetry run black --config pyproject.toml ./

.PHONY: formatting
formatting: codestyle

#* Linting
.PHONY: lint
lint: no-live mypy check-codestyle check-safety

.PHONY: test
test:
	poetry run pytest -c pyproject.toml
#	poetry run coverage-badge -o assets/images/coverage.svg -f

.PHONY: check-codestyle
check-codestyle:
	poetry run ruff --config=./pyproject.toml .
	poetry run black --config pyproject.toml ./
	poetry run darglint --verbosity 2 roc tests

.PHONY: mypy
mypy:
	poetry run mypy --config-file pyproject.toml ./

.PHONY: check-safety
check-safety:
	poetry check
	poetry run safety check --full-report -i 51457
	poetry run bandit -ll --recursive roc tests

.PHONY: update-dev-deps
update-dev-deps:
	poetry add -D bandit@latest darglint@latest mypy@latest pre-commit@latest pydocstyle@latest pylint@latest pytest@latest pyupgrade@latest safety@latest coverage@latest coverage-badge@latest pytest-html@latest pytest-cov@latest
	poetry add -D --allow-prereleases black@latest

.PHONY: no-live
no-live:
	grep '^\s*LIVE_DB\s*=\s*False\s*$$' tests/conftest.py
	grep '^\s*RECORD_DB\s*=\s*False\s*$$' tests/conftest.py

# Docs
.PHONY: update-api-docs
update-api-docs:
	poetry run sphinx-apidoc -f -o ./docs/_modules ./roc

.PHONY: docs
docs: update-api-docs
	cd docs && make html

.PHONY: edit-docs
edit-docs:
	poetry run sphinx-autobuild --pre-build "make update-api-docs" --host 0.0.0.0 --port 9000 docs docs/_build/html

# Coverage
.PHONY: coverage
coverage:
	poetry run pytest -c pyproject.toml --cov-report=lcov

.PHONY: doc-coverage
	poetry run interrogate

#* Docker
# Example: make docker-build VERSION=latest
# Example: make docker-build IMAGE=some_name VERSION=0.1.0
.PHONY: docker-build
docker-build:
	@echo Building docker $(IMAGE):$(VERSION) ...
	docker build \
		-t $(IMAGE):$(VERSION) . \
		-f ./docker/Dockerfile --no-cache

# Example: make docker-remove VERSION=latest
# Example: make docker-remove IMAGE=some_name VERSION=0.1.0
.PHONY: docker-remove
docker-remove:
	@echo Removing docker $(IMAGE):$(VERSION) ...
	docker rmi -f $(IMAGE):$(VERSION)

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: dsstore-remove
dsstore-remove:
	find . | grep -E ".DS_Store" | xargs rm -rf

.PHONY: mypycache-remove
mypycache-remove:
	find . | grep -E ".mypy_cache" | xargs rm -rf

.PHONY: ipynbcheckpoints-remove
ipynbcheckpoints-remove:
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

.PHONY: pytestcache-remove
pytestcache-remove:
	find . | grep -E ".pytest_cache" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

.PHONY: cleanup
cleanup: pycache-remove dsstore-remove mypycache-remove ipynbcheckpoints-remove pytestcache-remove

# commit hooks
.PHONY: pre-commit
pre-commit: lint

.PHONY: pre-push
pre-push: test
# docs doc-coverage coverage