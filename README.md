# rdvc-demo-project

This is a simple project to demonstrate [rDVC](https://github.com/exs-dmiketa/rdvc), a tool for executing DVC experiment pipelines on a SLURM cluster.

## Install

Clone the repository to a local directory and install it into a fresh virtual environment with

```bash
init_python_venv.sh
source .venv/bin/activate
```

## Run on a local machine

You can now run the project locally with

```bash
dvc exp run -S fabric.accelerator=cpu 
```
