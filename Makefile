.PHONY: lint-fix
lint-fix:
	poetry run ruff check --select I --fix .
	poetry run ruff format
	poetry run toml-sort -i pyproject.toml

.PHONY: test
test:
	poetry run pytest -vv .