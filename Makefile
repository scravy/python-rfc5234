lint:
	uv run ruff check
	uv run mypy .

test:
	uv run pytest

fmt:
	uv run ruff format

build:
	uv run python -m build

check: lint test build
	uv run twine check dist/*

publish: build twine-check
	uv run twine upload dist/*

clean:
	rm -rf dist/

.PHONY: lint test fmt build check publish clean
