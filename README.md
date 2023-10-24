# WRF-Ensembly

WRF-Ensembly is a tool for conducting Ensemble Assimilation experiments using the WRF (and WRF-CHEM) model and DART. It is implemented as a toolkit of Python scripts that handle the "gluing" between [WRF](https://github.com/wrf-model/wrf) and [DART](https://github.com/NCAR/DART), handling activities like running the necessary DART programs, creating the namelists, and converting observations using the appropriate converters, etc. Configuration for an experiment is handled through one config file to maximize reproducibility.

Running an experiment is done through a series of commands, each handling a separate task.