lint:
	uv run ruff check
	uv run mypy .

fmt:
	uv run ruff format

build:
	uv run python -m build

twine-check:
	uv run twine check dist/*

publish: build twine-check
	uv run twine upload dist/*

clean:
	rm -rf dist/
