# WRF Ensembly

WRF Ensembly is a toolkit for performing ensemble assimilation experiments with WRF(-CHEM) and NCAR DART. It provides all the 'glue' you need to use WRF and DART together. It is available as a Python-based command line tool and integrates with SLURM for use on HPCs. It also includes conveniences such as data management (both input and output data, as well as observation), pre- and post- processing, ensemble statistics and more! Everything is managed through a single config file.

The following features are available:

- WRF preprocessing (running all WPS steps and `real.exe`)
- Cycling with WRF and DART
- Handling WPS and WRF namelists
- Running WRF and DART
- Producing analysis files
- Producing ensemble statistics
- Applying custom post-processing scripts on forecasts and analysis files
- Doing everything in SLURM jobs
- Handling observation data (indexing and running DART's `obs_converters`)

## Project Status

This project is under active development. It is used at the National Observatory of Athens to conduct ensemble assimilation experiments both on desktop machines and on HPCs. Please come in contact if you find it useful!

## Quick Links

- [Getting Started](getting-started.md)
- [Installation Guide](user-guide/installation.md)
- [API Reference](api-reference.md)
- [GitHub Repository](https://github.com/yourusername/wrf_ensembly)
