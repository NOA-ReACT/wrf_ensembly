# Tutorial

This tutorial will walk you through a simple experiment using WRF-Ensembly. It will cover the basic steps of setting up an experiment, running it, and postprocessing the results.


## Prerequisites

You will need:
- A working installation of WRF-Ensembly. If you haven't installed it yet, follow the [Installation](./installation.md) guide.
- A working installation of WRF and WPS. You can find more information on how to install WRF [here](https://www2.mmm.ucar.edu/wrf/OnLineTutorial/compilation_tutorial.php). If you need it, you can use WRF-CHEM.
- A working installation of DART. You can find more information on how to install DART [here](https://dart.ucar.edu).

## Setting up the experiment

WRF-Ensembly keeps experiments inside a directory structure, which contains the model, any input and output files, and the configuration files. You can interact with experiments by putting the path as the first argument to the `wrf-ensembly` command.

```bash
wrf-ensembly /path/to/experiment group command [options]
```

All actions/commands are categorised into groups. You can read about all commands in [Usage](./usage.md). To begin, we will use the `experiment create` command to create a new experiment. This will create the directory structure and the configuration files needed to run the experiment.

```bash
wrf-ensembly /path/to/experiment experiment create iridium_chem_4.6.0
```

This will create a new experiment at `/path/to/experiment` using the config template `iridium_chem_4.6.0`. The templates are included in the [source code](https://github.com/NOA-ReACT/wrf_ensembly/tree/main/wrf_ensembly/config_templates).

Your first course of action should be to inspect and edit the `config.toml` file. This file contains all the configuration options for the experiment. You can find more information about the configuration options in the [Configuration](./configuration.md) section. The most important options to edit are:

- In the `[metadata]` section, set the experiment `name` and a small `description`.
- In the `[directories]` section, you must set the paths to `wrf_root`, `wps_root` and `dart_root`. For `wrf_root`, you should point to the directory that contains the `run/` directory of your WRF installation. For `wps_root`, you should point to the directory that contains the `geogrid.exe`, `ungrib.exe`, and `metgrid.exe` executables. For `dart_root`, you should point to the repository root (that contains `models/`, `obs/`, etc.)
- In the `[domain_control]` and `[time_control]` sections you can set up the relevant options from WRF. These options are passed to WRF and WPS when they are run.
- In the `[data]` section, you should set paths to `wps_geog`, `meteorology` (directory that contains the GRIB files), which `meteorology_vtable` you want to use (only filename, not a full path).
- In the `[assimilation]` section, set how many members your ensemble will have in `n_members`.

WRF-Chem users that want initial conditions w/ [interpolator-for-wrfchem](https://github.com/NOA-ReACT/interpolator_for_wrfchem) should also set the following variables in `[data]`:

```toml
manage_chem_ic = true
chemistry = { path = '/home/thgeorgiou/data/Modelling/AIRSENSE/CAMS_FC', model_name = 'cams_global_forecasts' }
```

Also read [WRF-CHEM](./wrf-chem.md) later for wrf-chem specific information.

After setting up your configuration, use the `experiment cycle-info` command to see the cycles of your experiment. This will show which cycles will run given your time control settings.

```bash
wrf-ensembly /path/to/experiment experiment cycle-info

# You can also use `--to-csv=` to save the table in a CSV file.
wrf-ensembly /path/to/experiment experiment cycle-info --to-csv=cycles
```

If everything looks good, you can use `experiment copy-model` to make a copy of WRF and WPS inside the experiment directory. This is required and is done to ensure you can run the experiment even if there are changes in the WRF or WPS installations.

```bash
wrf-ensembly /path/to/experiment experiment copy-model
```

## Preprocessing

At this stage, you should have a working experiment with the configuration set up. The next step is to preprocess the input data. This is done using the `preprocess` group of commands, which will run the necessary WPS steps to prepare the input data for WRF. The steps are:

```bash
# Prepare the preprocessing directory
wrf-ensembly /path/to/experiment preprocess setup

# Run the three main WPS steps
wrf-ensembly /path/to/experiment preprocess geogrid
wrf-ensembly /path/to/experiment preprocess ungrib
wrf-ensembly /path/to/experiment preprocess metgrid
```

The namelists for WPS are generated automatically based on the configuration file. All preprocessing takes place inside the `work/preprocess` subdirectory of the experiment.
The geogrid table is configurable in the `geogrid.table` configuration field, while the ungrib variable table is set in the `data.meteorology_vtable` configuration field.

At this point, you will be able to find the `met_em*` files inside the `work/preprocess/WPS` directory. To run real, you can use the `preprocess real --cycle <CYCLE>` command:

```bash
wrf-ensembly /path/to/experiment preprocess real --cycle 0
```

The above command will take care to generate the namelist for `real.exe`, run `real.exe`, and copy the final `wrfinput_d01` and `wrfbdy_d01` files to the `data/initial_conditions` directory. You must run real for every cycle in your experiment.

To make this whole process easier, you can generate a SLURM jobfile for preprocessing using the `slurm preprocessing` command:

```bash
wrf-ensembly /path/to/experiment slurm preprocessing
sbatch /path/to/experiment/jobfiles/preprocessing.sh
```

You can setup which `#SBATCH` directives you want to include in the jobfile by editing the `[slurm]` section of the config file.

Successful execution of all preprocessing steps will result in the `data/initial_conditions` directory being populated with the `wrfinput_d01` and `wrfbdy_d01` files for each cycle. You can check the status of the preprocessing using the `status` command:

If you are using WRF-Chem and you want to use the interpolator-for-wrfchem to generate initial conditions, you can use the `preprocess interpolate-chem` command:

```bash
wrf-ensembly /path/to/experiment preprocess interpolate-chem
```

## Observations

Every data assimilation experiment needs observations to assimilate. DART reads observations in its `obs_seq` format, but WRF-Ensembly keeps its own observation database per experiment, so the same observations can also be used for plotting and validation. The [Observations](observations.md) page covers this in detail. The workflow is:

```mermaid
flowchart TD
A1[EarthCARE EBD<br/>ec_atl_ebd.h5] --> B[wrf-ensembly-obs convert]
A2[AERONET<br/>*.lev20] --> B
A3[MODIS AOD<br/>modis.hdf] --> B
B --> C[Standardised .parquet files]
C --> D[observations add<br/>trimming, superobbing]
D --> E[(obs/observations.duckdb)]
E --> F[observations prepare-cycles]
F --> G[obs/cycle_NNN.obs_seq]

classDef inputFiles fill:#e1f5fe,stroke:#01579b,stroke-width:2px
classDef tools fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
classDef final fill:#ffebee,stroke:#c62828,stroke-width:2px

class A1,A2,A3 inputFiles
class B,D,F tools
class G final
```

First, convert the raw instrument files to the standardised parquet format using the `wrf-ensembly-obs` CLI. It works on files, not on an experiment, so you can convert once and reuse the output across experiments:

```bash
wrf-ensembly-obs convert aeronet input_file.lev20 aeronet.parquet --quantities AOD_500nm
```

Then add the converted files to the experiment. This trims them to the domain and the experiment's time range, applies any density reduction (superobbing, binning, thinning) configured in `config.toml`, and stores them in `obs/observations.duckdb`:

```bash
wrf-ensembly /path/to/experiment observations add /path/to/observations/*.parquet --jobs 4
wrf-ensembly /path/to/experiment observations show
```

Finally, extract the observations of each cycle's assimilation window and convert them to `obs_seq`. This requires the `wrf_ensembly` observation converter to be compiled in DART (`DART/observations/obs_converters/wrf_ensembly`):

```bash
wrf-ensembly /path/to/experiment observations prepare-cycles --jobs 8
```

This writes one `cycle_NNN.obs_seq` file per cycle (plus a `cycle_NNN.parquet` for inspection) in the `obs/` directory. If a cycle has no file, no observations fall inside its assimilation window, and nothing will be assimilated for it. Use `observations cycle-summary` to see the counts per cycle.


## Preparing the ensemble

Now that you have the initial conditions and the observations ready, you can run the ensemble. This is done using the `ensemble` group of commands, which handle preparing the ensemble, advancing the members, running the assimilation filter and finally cycling. The commands are shown visually in the diagram below:

```mermaid
flowchart TD
    setup[ensemble setup] --> genperts[ensemble generate-perturbations]
    genperts --> applyperts[ensemble apply-perturbations]
    applyperts --> updatebc[ensemble update-bc]
    updatebc --> advance[ensemble advance-member]
    advance --> filter[ensemble filter]
    filter --> analysis[ensemble analysis]
    analysis --> cycle[ensemble cycle]
    cycle --> updatebc
```

We begin with the `ensemble setup` command, which prepares the ensemble by copying the initial conditions and setting up the directories for each member. This command should be run only once at the beginning of the experiment.

```bash
wrf-ensembly /path/to/experiment ensemble setup
```

Now is the time to handle perturbations, if you are using them. Perturbations are used to generate an ensemble of members that are slightly different from each other from one set of initial conditions. First, you must add the appropriate perturbation configuration in the `config.toml` file, under the `[perturbations]` section. You can read more about perturbations in the [Configuration](./configuration.md) section. For example, if we want to perturb the U and V fields:

```toml
[perturbations.variables.V]
operation = 'add'
mean = 0
sd = 8
rounds = 8
boundary = 10

[perturbations.variables.U]
operation = 'add'
mean = 0
sd = 8
rounds = 8
boundary = 10
```

After setting up the perturbations, you can generate them using the `ensemble generate-perturbations` command:

```bash
wrf-ensembly /path/to/experiment ensemble generate-perturbations --jobs 8
```

This will generate the perturbation files inside `data/diagnostics/perturbations/`. You can inspect the files to see how the perturbations look like. If you want to apply the perturbations to the initial conditions, you can use the `ensemble apply-perturbations` command:

```bash
wrf-ensembly /path/to/experiment ensemble apply-perturbations --jobs 8
```

You can repeat the `setup` -> perturbations process as many times as you want to tune your perturbations. The `setup` command will always copy the original `wrfinput_d01` and `wrfbdy_d01` files from the `data/initial_conditions/` directory, so you can always start fresh.

The `update-bc` step is crucial after applying perturbations or cycling. When you modify the initial conditions field, you might introduce inconsistencies with the boundary conditions if there are changes near the boundary. These inconsistencies can lead to unexpected behavior in the model. The `ensemble update-bc` will ensure that the edges of the domain are consistent with the boundary conditions using the `update-wrf-bc` tool from DART. You should run this command after applying perturbations or cycling the ensemble:

```bash
wrf-ensembly /path/to/experiment ensemble update-bc
```


## Running WRF

It is finally time to advance the model, which is done using the `ensemble advance-member` command:

```bash
# Advance member 2 to the next cycle using 24 cores
wrf-ensembly /path/to/experiment ensemble advance-member --member 2 --cores 24
```

The forecasts are stored inside `scratch/forecasts/cycle_ABC`. Of course, you must advance all members to the next cycle before continuing. There is a SLURM helper for this we will cover later.


## Running the assimilation filter, generating the analysis

Running the assimilation filter involves placing the model state files in the correct place and running `filter.exe` from the DART WRF directory. Currently, WRF-Ensembly does not handle the DART namelist, so you might have to make some changes to `input.nml` in the `models/wrf/work` directory. Namely, you should set the correct number of ensemble members in `filter_nml::ens_size` and the correct observation types in `obs_kind_nml::assimilate_these_obs_types`. The state variables in `model_nml::wrf_state_variables` must also match `config.yml` and what you expect.

After adjusting `input.nml`, you can run the assimilation filter using the `ensemble filter` command:

```bash
wrf-ensembly /path/to/experiment ensemble filter
```

The DART output files are stored in `scratch/dart/cycle_ABC`. After filter is executed, this command will automatically move the `obs_seq.final` file to `data/diagnostics/cycle_ABC.obs_seq.final` so you can check how the assimilation went later.

At this point, you can generate the analysis files, which are the final forecast `wrfout` files but with the fields corrected by the assimilation. This is done using the `ensemble analysis` command:

```bash
wrf-ensembly /path/to/experiment ensemble analysis
```

The analysis files are stored in `scratch/analysis/cycle_ABC`.


## Cycling the experiment

Finally, you can cycle the experiment using the `ensemble cycle` command. This will prepare the members for the next cycle by copying the new initial and boundary condition files, and adding the new analysis fields to them. After `cycle`, you should run `update-bc` and then the members are ready to be advanced to the next cycle. You can run the `cycle` command as follows:

```bash
wrf-ensembly /path/to/experiment ensemble cycle
wrf-ensembly /path/to/experiment ensemble update-bc
```

## Automating all this with SLURM

Running all the above steps is very tedious if you have more than a toy amount of cycles and members. WRF-ensembly provides a command to queue the whole experiment with SLURM, so you can run it in the background and forget about it. The command is `slurm run-experiment`, which will generate a set of jobs to run your experiment.

```bash
wrf-ensembly /path/to/experiment slurm run-experiment
```

Specifically, you will get N+1 jobs, where N is the number of members. There is one job per member to advance and there is a final job that runs the assimilation filter and cycles the experiment. The jobs are automatically submitted to SLURM and the final one uses dependencies to ensure that it runs only after all members have been advanced. A side-effect for this is that you must have permission to submit N+1 jobs to the SLURM queue, potentially limiting your ensemble size.