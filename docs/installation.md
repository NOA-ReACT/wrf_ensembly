# Installation

WRF-Ensembly is available on PyPI. The minimum supported Python version is 3.11. You can install it using pip:

```bash
pip install wrf_ensembly
```

It's recommended to install it in a virtual environment. You can either use `venv` or a conda derivative. For example:

```bash
python -m venv ./venv
source ./venv/bin/activate
pip install wrf_ensembly

# Or with conda
conda create -n wrf_ensembly python=3.11
conda activate wrf_ensembly
pip install wrf_ensembly

# Or mamba
mamba create -n wrf_ensembly python=3.11
mamba activate wrf_ensembly
pip install wrf_ensembly
```

## External dependencies

WRF-Ensembly requires the following external dependencies:

- Compiled WRF and WPS. Version is not important but the latest is recommended and what is used by the developers. More info on the model [here](https://github.com/wrf-model/WRF).
- Compiled DART. Again version is not important but all tests are done with the latest github clone. More info on DART [here](https://github.com/NCAR/DART).
- For postprocessing, `cdo` and `nco` are used to wrangle with the netCDF files. If you installed WRF-Ensembly in a conda/mamba environment, you can also install `cdo` and `nco` with:

```bash
# or mamba or micromamba or ...
conda install -c conda-forge cdo nco
```

## Development & Contributing

If you want to contribute to WRF-Ensembly, you can clone the repository and install it in editable mode. This allows you to make changes to the code and test them without reinstalling the package. We use [uv](https://docs.astral.sh/uv/) for dependency management and packaging. After installing uv, clone the repository and install the dependencies:

```bash
git clone https://github.com/NOA-ReACT/wrf_ensembly.git
cd wrf_ensembly
uv sync
```

uv will create a virtual environment for you that contains WRF-Ensembly in editable mode. Any changes you make in the repository should apply to the `wrf-ensembly` command.

## Read next

You can either go through the [Usage](./usage.md) section to learn how to use WRF-Ensembly, or you can read about the [Core concepts](./core-concepts.md) to understand how it works. There is also a [Tutorial](./tutorial.md) that walks you through a simple experiment.