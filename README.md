# SoNetSim

> A **So**cial **Net**work **Sim**ulator Based on Large Language Model Agents.

This project can be roughly (the boundary is not that clear) devided into two
parts:

- "Frontend": [SoNetSim](https://github.com/tongyx361/sonetsim)
- "Backend": [OASIS (Fork)](https://github.com/tongyx361/oasis)

The frontend is based on Streamlit. Please check
[their website](https://streamlit.io/) for more details.

The backend is a fork of the OASIS project from CAMEL-AI. We fork for bug
resolving, feature customizing and performance optimizing. Please check
[their repo](https://github.com/camel-ai/oasis) for more details.

## Setup

First clone the repo with the `oasis` submodule:

```bash
git clone https://github.com/tongyx361/social-network-simulator.git --recurse-submodules
cd social-network-simulator
```

We recommend to use `conda` (or more efficient `micromamba`) to manage the
Python environment:

```bash
conda create -n sonetsim python=3.11 # micromamba create -n sonetsim python=3.11
conda activate sonetsim # micromamba activate sonetsim
```

We recommend to use `pip` (or more efficient `uv`) to manage the Python
packages:

```bash
pip install -e "./oasis" # uv pip install -e "./oasis"
pip install -e ".[frontend]" # uv pip install -e ".[frontend]"
# For development, add the `dev` extra
# pip install -e ".[frontend,dev]" # uv pip install -e ".[frontend,dev]"
```

Then just run the demo app with:

```bash
streamlit run src/examples/demo.py
```

## Contribution Guide

See [CONTRIBUTING.md](.github/CONTRIBUTING.md).
