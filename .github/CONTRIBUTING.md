# Contribution Guide

1. Please try to make pull requests instead of pushing to the `main` branch
   directly. (But it's also fine.)
2. For any questions, please contact @tongyx361.

## Setup

First setup the environment as described in the [README](../README.md#setup).

We use `pre-commit` to manage code quality checks. Please install it with:

```bash
pre-commit install
```

Then every time you run `git commit`, `pre-commit` will check the code with the
hooks defined in [`.pre-commit-config.yaml`](../.pre-commit-config.yaml).

Then you need to fix the issues and run `git add` and `git commit` again until
the checks pass.

Note that some errors can be automatically fixed (e.g. `ruff`), so sometimes you
just need to add the changes and commit again.

You can also run `pre-commit run [hook-id(s)] [--all-files]` to manually run the
checks (optionally with specified ones or on all the files).

If you really want to skip the checks, you can use `git commit --no-verify`.

## Repository Structure

The core parts of the repository are as follows:

- [`oasis`](../oasis): the "backend" submodule, which can be treated as an
  independent repository.
- [`src`](../src): the Python source code (for "frontend"), which will be
  checked by `mypy` as defined in
  [`.pre-commit-config.yaml`](../.pre-commit-config.yaml).
  - [`sns`](../src/sns): the "Social Network Simulator" package code.
  - [`examples`](../src/examples): example code including the demo app.
  - [`notebooks`](../src/notebooks): `ipynb` files.
- [`data`](../data): the data home.
  - [`agent_info`](../data/agent_info): the agent information home, containing
    `csv`s of the agent information.
  - [`simu_db`](../data/simu_db): the default destination to save the simulation
    results.
  - [`visualization`](../data/visualization): the default destination to save
    the visualization results.

The auxiliary parts of the repository are as follows:

- [`tests`](../tests): the test code, which will be checked by `mypy` as defined
  in [`.pre-commit-config.yaml`](../.pre-commit-config.yaml).
- [`.pre-commit-config.yaml`](../.pre-commit-config.yaml): the configuration for
  `pre-commit`.
- [`pyproject.toml`](../pyproject.toml): the configuration for the Python
  project, used for modern Python tool configuration like for `ruff` and `mypy`.
- ...
