# Social Network Simulator

> A Social Network Simulator Based on Large Language Model Agents.

## Introduction

This project can be roughly (the boundary is not that clear) devided into two
parts:

- "Frontend": https://github.com/tongyx361/social-network-simulator
- "Backend": https://github.com/tongyx361/oasis

The frontend is based on Streamlit. Please check
[there website](https://streamlit.io/) for more details.

The backend is a fork of the OASIS project from CAMEL-AI since there exist some
bugs and we also need some customizations and optimizations. Please check
[their repo](https://github.com/camel-ai/oasis) for more details.

## Setup

We recommend to use `conda` (or more efficient `micromamba`) to manage the
Python environment:

```bash
conda create -n sns python=3.11 # micromamba create -n sns python=3.11
conda activate sns # micromamba activate sns
```

We recommend to use `pip` (or more efficient `uv`) to manage the Python
packages:

```bash
git clone https://github.com/tongyx361/social-network-simulator.git --recurse-submodules
cd social-network-simulator
cd oasis
# pip install uv
pip install -e "." # uv pip install -e "."
cd ..
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
