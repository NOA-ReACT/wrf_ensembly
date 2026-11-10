# WRF Ensembly

WRF Ensembly is a toolkit for performing ensemble assimilation experiments with NCAR DART and a limited-area model: [WRF](wrf.md) (and [WRF-Chem](wrf-chem.md)) or [ICON](icon.md). It provides all the 'glue' you need to use the model and DART together. It is available as a Python-based command line tool and integrates with SLURM for use on HPCs. It also includes conveniences such as data management (both input and output data, as well as observation), pre- and post- processing, ensemble statistics and more! Everything is managed through a single config file.

The following features are available:

- Preprocessing of initial and boundary conditions (WRF: all WPS steps and `real.exe`; ICON: IFS data to the ICON grid)
- Cycling with the model and DART
- Handling the namelists of the model and of DART
- Running the model and DART
- Producing analysis files
- Producing ensemble statistics
- Applying custom post-processing scripts on forecasts and analysis files
- Doing everything in SLURM jobs
- Handling observation data (converting, indexing and writing DART's `obs_seq` files)

The name comes from WRF, the first model that was supported. The workflow is the same for both models, see [Core Concepts](core-concepts.md) for what differs.

## Project Status

This project is under active development. It is used at the National Observatory of Athens to conduct ensemble assimilation experiments both on desktop machines and on HPCs. Please come in contact if you find it useful!

## Quick Links

- [Core Concepts](core-concepts.md)
- [Tutorial](tutorial.md) (with WRF)
- Model pages: [WRF](wrf.md), [ICON](icon.md)
- [Installation Guide](installation.md)
- [Command Reference](usage.md)
- [GitHub Repository](https://github.com/NOA-ReACT/wrf_ensembly)
