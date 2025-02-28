# template-project

## What to do at first

At first, execute in terminal:

    python -m venv .venv

Then, activate the virtual environment:

    source .venv/bin/activate

Each time a new package is installed include it to the ```requirements.txt```.


## Sanity Checks

These instructions are executed in each push, but if you want to check code and typing style before pushing please follow these steps:

    pytest
    black --check .
    ruff check
    mypy src tests
    flake8 src tests
    complexipy .
    pylint --fail-under=8 src tests

These instructions are not automatically executed because sometimes they might be too restrictive, but is a good practice to check periodically if a 10/10 mark is achieved in **Pylint** and a 100% test coverage is achieved in **coverage**:

    pylint src tests
    pytest --cov-report term-missing


## Others

To see the documentation run:

    mkdocs serve
