.DEFAULT_GOAL := help
PY ?= python3
VENV := .venv
BIN := $(VENV)/bin
TORCH_CPU := --extra-index-url https://download.pytorch.org/whl/cpu

.PHONY: help venv install install-dev install-storage model dev openapi run test lint fmt typecheck docker-build docker-up docker-down clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n",$$1,$$2}'

venv:  ## Create the virtualenv
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -U pip wheel

install: venv  ## Install runtime deps (CPU torch)
	$(BIN)/pip install -r requirements.txt $(TORCH_CPU)

install-dev: venv  ## Install runtime + dev deps
	$(BIN)/pip install -r requirements-dev.txt $(TORCH_CPU)

install-storage: venv  ## Install runtime + boto3
	$(BIN)/pip install -r requirements-storage.txt $(TORCH_CPU)

model:  ## Pre-download the model weights
	$(BIN)/python scripts/download_model.py

dev:  ## Run with autoreload, no auth, /docs on (for Postman)
	RMBG_ENVIRONMENT=development RMBG_LOG_JSON=false RMBG_API_KEYS= \
	RMBG_ENABLE_DOCS=true RMBG_RATE_LIMIT_ENABLED=false \
	$(BIN)/uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

openapi:  ## Export the OpenAPI schema to postman/openapi.json
	$(BIN)/python scripts/export_openapi.py

run:  ## Run the production server
	$(BIN)/gunicorn app.main:app -c deploy/gunicorn.conf.py

test:  ## Run the test suite
	$(BIN)/pytest -q

lint:  ## Lint
	$(BIN)/ruff check app scripts tests

fmt:  ## Autofix + format
	$(BIN)/ruff check --fix app scripts tests
	$(BIN)/ruff format app scripts tests

typecheck:  ## Static types
	$(BIN)/mypy app

docker-build:  ## Build the image
	docker compose -f deploy/docker-compose.yml build

docker-up:  ## Start the stack
	docker compose -f deploy/docker-compose.yml up -d

docker-down:  ## Stop the stack
	docker compose -f deploy/docker-compose.yml down

clean:  ## Remove caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache
