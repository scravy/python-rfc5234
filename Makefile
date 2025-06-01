check: fmt lint test nox build
	uv run twine check dist/*

lint:
	uv run ruff check
	uv run mypy .

test:
	uv run pytest -n auto

nox:
	uv run nox

fmt:
	uv run ruff format

build:
	uv run python -m build

publish: build twine-check
	uv run twine upload dist/*

clean:
	rm -rf dist/

.PHONY: lint test fmt build check publish clean
