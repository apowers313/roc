#* Variables
SHELL := /usr/bin/env bash
PYTHON := python
PYTHONPATH := `pwd`

#* Docker variables
IMAGE := roc
VERSION := latest

# Start unified server + Vite dev frontend
.PHONY: run
run:
	@npx servherd start -n roc-server -p 9043 -e roc_dashboard_port={{port}} -- uv run server --port {{port}} >/dev/null
	@npx servherd start -n roc-ui -p 9044 -e 'VITE_API_PORT={{$$ "roc-server" "port"}}' -e VITE_DEV_PORT={{port}} -e VITE_HOST={{hostname}} -- pnpm -C dashboard-ui dev >/dev/null
	@echo "Dashboard: $$(npx servherd info roc-ui --json 2>/dev/null | jq -r .data.url)"

# Stop all ROC servers
.PHONY: stop
stop:
	@npx servherd stop roc-ui 2>/dev/null || true
	@npx servherd stop roc-server 2>/dev/null || true
	@echo "Servers stopped."

# Run game directly (legacy, no server management)
.PHONY: play
play:
	uv run play

#* Poetry
.PHONY: uv-download
uv-download:
	curl -LsSf https://astral.sh/uv/install.sh | sh

.PHONY: uv-remove
uv-remove:
	uv cache clean
	rm -r "$(uv python dir)"
	rm -r "$(uv tool dir)"
	rm ~/.local/bin/uv ~/.local/bin/uvx

#* Installation
.PHONY: setup
setup: install pre-commit-install

.PHONY: install
install:
	uv venv --python 3.13
	uv sync
	uv run mypy --install-types --non-interactive ./

.PHONY: pre-commit-install
pre-commit-install:
	uv run pre-commit install --hook-type pre-commit --hook-type pre-push

#* Formatters
.PHONY: codestyle
codestyle:
	uv run ruff format --config pyproject.toml ./

.PHONY: formatting
formatting: codestyle

#* Linting
.PHONY: lint
lint: mypy check-codestyle dep-check

.PHONY: dep-check
dep-check:
	uv run deptry .

.PHONY: load-data
load-data:
	mgconsole < tests/data/got.cypherl

.PHONY: test
test:
	uv run pytest -c pyproject.toml

.PHONY: test-unit
test-unit:
	uv run pytest -c pyproject.toml tests/unit/ -q

.PHONY: test-integration
test-integration:
	uv run pytest -c pyproject.toml tests/integration/ -q

.PHONY: test-e2e
test-e2e:
	uv run pytest -c pyproject.toml tests/e2e/ -q

.PHONY: profile
profile:
	# for example:
	# make TEST=tests/flood_test.py profile
	# or
	# make TEST="tests/attention_test.py -k test_basic" profile
	uv run pytest -c pyproject.toml --profile-svg $(TEST)

.PHONY: check-codestyle
check-codestyle:
	uv run ruff check --config=./pyproject.toml .
	uv run ruff format --check --config pyproject.toml ./
	uv run ruff check --config ./pyproject.toml --fix

.PHONY: mypy
mypy:
	uv run mypy --config-file pyproject.toml ./

# safety check exceptions:
# 51457, 70612: Py 1.11.0 Regexp DoS - not processing external data
# 67599: Pip --extra-index-url - not using --extra-index-url
# 72715: MKDocs Material RXSS vulnerability - not publishing docs in a way where
# XSS matters
# 73456: virtualenv package are vulnerable to command injection - temporarily
# disable until "poetry update" is fixed

.PHONY: check-safety
check-safety:
	uv sync
	uv run safety check --full-report -i 51457 -i 67599 -i 70612 -i 72715
	uv run bandit -ll --recursive roc tests

.PHONY: update-dev-deps
update-dev-deps:
	uv add -D bandit@latest mypy@latest pre-commit@latest pydocstyle@latest pylint@latest pytest@latest pyupgrade@latest safety@latest coverage@latest coverage-badge@latest pytest-html@latest pytest-cov@latest ruff@latest

# Docs
.PHONY: docs
docs:
	uv run mkdocs build --clean

.PHONY: edit-docs
edit-docs:
	uv run mkdocs serve -a 0.0.0.0:9000 -w roc

# Coverage
.PHONY: coverage
coverage:
	uv run coverage run -m pytest
	uv run coverage lcov
	uv run coverage xml
	uv run coverage report

.PHONY: coverage-server
coverage-server:
	cd htmlcov && python3 -m http.server 9099

# SonarQube
.PHONY: sonar-reports
sonar-reports:
	uv run ruff check --config=./pyproject.toml . --output-format json --output-file ruff-report.json --exit-zero
	[ -s ruff-report.json ] || echo '[]' > ruff-report.json
	uv run mypy --config-file pyproject.toml ./ --junit-xml mypy-report.xml || true
	cd dashboard-ui && npx vitest run --coverage || true

.PHONY: sonar
sonar: sonar-reports
	@if [ -z "$$SONAR_TOKEN" ]; then echo "ERROR: SONAR_TOKEN environment variable is not set. Export it before pushing." >&2; exit 1; else npx @sonar/scan; fi

.PHONY: doc-coverage
doc-coverage:
	uv run interrogate -c pyproject.toml -vv roc

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

.PHONY: clean
clean:
	git clean -fxd

# commit hooks
.PHONY: pre-commit
pre-commit: 

.PHONY: pre-push
pre-push: check-safety lint test-unit doc-coverage coverage sonar docs

.PHONY: upgrade-python
upgrade-python:
	uv run pyupgrade `find roc -name "*.py" -type f`
	uv run pyupgrade `find tests -name "*.py" -type f`