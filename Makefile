.PHONY: dev start lint lint-fix format
DIR_CACHE = ".pycache"
ENV_PYCACHE = PYTHONPYCACHEPREFIX=$(DIR_CACHE)

PORT ?= 8000

dev:
	$(ENV_PYCACHE) uv run fastapi dev --port $(PORT)

start:
	uv run fastapi run --port $(PORT)

lint:
	uv run ruff check

lint-fix:
	uv run ruff check --fix

format:
	uv run ruff format

install:
	uv sync && uv run pre-commit install
