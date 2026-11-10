# Core Concepts

WRF ensembly is made available as a command line tool, which can be accessed with `wrf-ensembly` after installation. All activities are managed through **Experiments**, which are represented by a *directory* that holds all configuration, data and model binaries needed for one assimilation experiment. In general, all commands follow the pattern:

```bash
wrf-ensembly $EXPERIMENT_PATH group command
```

where `$EXPERIMENT_PATH` is the path to the experiment directory and `group`/`command` signals the action you want to execute.

Experiments follow the general 'Setup, Preprocessing, Assimilation, Postprocessing' steps. Each step is managed through a separate group of commands.


## Cycling

WRF Ensembly is a cycling assimilation system based on [NCAR DART](https://dart.ucar.edu). The model is [WRF](./wrf.md) (or WRF-Chem) or [ICON](./icon.md), chosen with `metadata.model` in the configuration. With *cycling*, we refer to the act of running a forward run of the model and then using its output as input for the next forward run. One *cycle* of an *experiment* refers to one forward run and the subsequent assimilation.

```mermaid
flowchart LR
  nwp[Model] -->|forward run| forecast[("Forecast")]
  forecast --> dart[DART]
  dart -->|Assimilation| analysis[("Analysis")]
  analysis -->|Start of the next cycle| nwp

  observations[("Observations")] --> dart
```

Since the system is an **Ensemble** assimilation system, the forward run is actually a set of forward runs, one for each member of the ensemble. Then, DART is used to produce the analysis (for all members) and you can optionally use ensembly to get ensemble statistics (e.g. mean, spread).

Optionally, a cycle can use the forecast to move forward, in case the analysis doesn't
exist. For example, when there aren't any data to assimilate.

### What is the same for both models and what is not

The experiment directory, the commands of a cycle (`advance-member`, `filter`, `analysis`, `cycle`), the status tracking, the observations, the SLURM jobs and the layout of the postprocessed output are the same for WRF and ICON. What differs is where the initial and boundary conditions come from and how a member continues from the analysis:

| | WRF, `wrfinput` mode | WRF, `restart` mode | ICON |
| --- | --- | --- | --- |
| Initial and boundary conditions | WPS and real.exe, for every cycle | WPS and real.exe, once | IFS data interpolated to the grid, once |
| The filter reads the prior from | wrfout at the end of the cycle | wrfout at the end of the cycle | First guess file at the end of the cycle |
| The next cycle starts from | A fresh wrfinput with the analysis' `cycled_variables` | The member's restart file with the analysis' `state_variables` | The first guess and an analysis file, combined by ICON |
| Boundaries after the analysis | `update-bc` | `update-bc` (increment only) | Nothing to do |
| Several cycles in one run ([segments](./wrf.md#segments)) | No | Yes | No |
| `[perturbations]` | Yes | Yes | First cycle only, see [Perturbations](./icon.md#perturbations) |

The details are on the page of each model: [WRF](./wrf.md), [ICON](./icon.md).

## Experiment Directory Structure

The experiment directory is the main directory where all data and configuration files are stored. It is structured as follows:


  - `data/` - Input files & final output files
    - `analysis/`
    - `diagnostics/` - DART obs_seq.out files, IC/BC perturbations
    - `forecasts/` - Postprocessed model output
    - `initial_boundary/` - Initial and boundary conditions (WRF: wrfinput and wrfbdy files)
  - `jobfiles/` - SLURM jobfiles
  - `logs/` - Log files, one subdirectory per executed command
    - `slurm/` - stdout of SLURM jobs
  - `obs/` - Observations (.toml and .obs_seq)
  - `scratch/` - Temporary files (but also the raw model output)
    - `analysis/`
    - `dart/` - DART output files (netCDF only w/ state vars.)
    - `forecasts/` - raw output files from the model (wrfout, or ICON's `icon_*.nc` and first guess files)
    - `postprocess/` - Used during post-processing
    - `restart/` - WRF restart files per cycle and member (`restart` cycling mode), and
      the checkpoints of a running segment with `checkpoints.json`
  - `work/` - Model executables and working directories
    - `ensemble/` - One subdirectory per member, the model runs in it
    - `preprocessing/` - WRF only: a copy of WRF and WPS used to generate the IC/BC
    - `WPS/` - WRF only: the WPS build used for the experiment
    - `WRF/` - WRF only: the WRF build used for the experiment
  - `status/` - Experiment status, see [Experiment status tracking](#experiment-status-tracking)
  - `config.toml` - Configuration file for the experiment

The `scratch/` directory can get pretty large in size because of the raw model output. You can set move this directory outside of the experiment path, possibly on a different mountpoint. This is to accomodate HPC systems that provide a larger scratch mountpoint.


## Logging

Everytime a command is executed, a subdirectory is created inside the `logs/` directory, following this naming scheme:

```
YYYY-MM-DD_HHMMSS-COMMAND
```

For example, `2024-12-31_235959-experiment-create` would be created when running `experiment create` on 2024-12-31, at 23:59:59. Inside the subdirectory, there is always a file called `wrf_ensembly.log` that contains all output of ensembly. If any external commands were used (i.e. `ungrib.exe`), then the output of the command in stored in a second file (i.e. `ungrib.log`).

There is a special case for the `rsl.*` files created by WRF. Because you have 2 files per CPU core and you create new files every cycle, for each member, it can easily add up and hit file-count quotas on certain HPC systems. Thus, only the first rsl files are available in plain-text (`rsl.out.0000` and `rsl.error.0000`). The rest of the files are compressed and stored in the same directory.


## Observations

Observations are stored inside the `obs/` category per-experiment. DART requires observations to be in `obs_seq` format and provides a set of `obs_converters` to do this. Since you would need one `obs_seq` file per assimilation cycle, including all observations you need (possibly from different instruments), WRF-Ensembly tries to automate this process by having scripts to browse available observations and then convert only the required ones. This is a per-observation-type workflow and might need the user to write a custom script to find available observations.

The whole abstraction is entirely optional. WRF-Ensembly will use any `obs_seq.cycle_X` available inside the `obs/` directory. More information in the [Observations](./observations.md) section.


## Configuration

Experiments are configured through the `config.toml` file, in [TOML format](https://toml.io/en/). This includes all WRF-Ensembly settings, the namelists of the model (WRF and WPS, or ICON) and the DART namelist. More information in the [Configuration](./configuration.md) section.


## Postprocessing

In the typical WRF-Ensembly workflow, we don't use the raw model output. The following steps are followed before you get the output **forecast**/**analysis** files:

1. Bring the files to a common layout: dimensions `t, z, y, x`, CF-compliant names and attributes, some diagnostics computed. For WRF this is done with [xwrf](https://github.com/xarray-contrib/xwrf); ICON output is interpolated from its triangular grid to a regular lat/lon grid. Remove any uneeded variables to save space.
2. Optionally, apply any external processors on the files.
3. Concatenate all files into a single file, one per cycle and member.
4. Compute the ensemble statistics (mean and standard deviation), leaving you with two files per cycle.

During the final steps, the files are also compressed using a netCDF-compatible algorithm (zstd is recommended, but zlib is more widely available) and the variables are 'packed' to avoid storing too many insignificant digits. Both these processes are optional and configurable (see the `[postprocess]` section of the [Configuration](./configuration.md#postprocess)). The user can select whether to store only the ensemble statistics or all member files, the first cutting down the output size significantly.

Because there are many tools that use the raw output files, the above is all optional. You can find them inside the `scratch/` directory.

More information about postprocessing is available in the [Postprocessing](./postprocess.md) section, including how the processing pipeline is setup.


## HPC intergration (SLURM)

For use in HPC environments, WRF-Ensembly heavily relies on SLURM for job scheduling. This is an optional convenience feature, as you can completely use WRF-Ensembly without a scheduling system.
Jobfiles can be generated for many time-intensive tasks (i.e. pre-processing or postprocessing with WRF-Ensembly) and for running the assimilation experiment. In this last scenario, WRF-ensembly uses dependencies to run the assimilation/cycle step after the model has advanced, which then queues the next cycle. The example below shows a flowchart for an experiment with 4 members. Each box represents a SLURM job.

```mermaid
flowchart TB
advance_c0_1[Run model member #1] --> filter_1[Assimilation/cycling for cycle 1]
advance_c0_2[Run model member #2] --> filter_1
advance_c0_3[Run model member #3] --> filter_1
advance_c0_4[Run model member #4] --> filter_1

filter_1 --> advance_c1_1[Run model member #1]
filter_1 --> advance_c1_2[Run model member #2]
filter_1 --> advance_c1_3[Run model member #3]
filter_1 --> advance_c1_4[Run model member #4]

advance_c1_1 --> filter_2[Assimilation/cycling for cycle 2]
advance_c1_2 --> filter_2
advance_c1_3 --> filter_2
advance_c1_4 --> filter_2

filter_2 --> advance_c2_1[Run model member #1]
filter_2 --> advance_c2_2[Run model member #2]
filter_2 --> advance_c2_3[Run model member #3]
filter_2 --> advance_c2_4[Run model member #4]
```

This feature allows the user to run experiments fully parallelized, with each model run taking place in a different node (as long as there are available resources), while only requiring N+2 queued jobs at any time, where N is the number of members. This is useful in systems where there are max-queued job limits.

If for any reason the experiment is interrupted, it can be resumed without re-running any completed steps.

With [segments](./wrf.md#segments), each "Run model member" job runs a whole segment, and its
time limit follows the segment's length.


## Experiment status tracking

The status of the experiment is tracked through the `status/` directory, which stores the current cycle, which members have been advanced (i.e. the model successfully completed), whether the assimilation filter has been completed, and some statistics about how long it took to advance each member. There is a set of commands available for working with it (`wrf-ensembly status`), described in the [Status](./usage.md#status) section.

The layout is one small file per fact:

```
status/
  experiment.json                       # {"version": 1, "current_cycle": 5}
  cycles/cycle_000/
    members/member_00.json              # advancement and runtime of one member
    filter_complete                     # marker files; existence means done
    analysis_complete
    cycle_complete
    ops/apply_perturbations             # optional operation markers
  segments/cycle_004.json               # plan of the segment starting at cycle 4
```

Every file has exactly one writer and is written atomically, so no locking is involved anywhere. This matters because ensemble members advance as separate jobs on separate nodes, often on a shared filesystem where the locking that a database needs is unreliable or slow. Members writing their own files in parallel cannot collide, and everything else is written by a single serial command.

The state of a cycle is not stored directly, it is worked out from these files: a cycle with no member files is `initialized`, one with some is `advancing_members`, one where every member has a file is `members_advanced`, and beyond that the marker files take over. This means the status can never disagree with itself, and a member finishing in another process is picked up immediately.

Everything is plain text, so you can inspect an experiment with `ls` and fix one by hand:

```bash
# How far along is cycle 12?
ls $EXPERIMENT/status/cycles/cycle_012/members | wc -l

# Re-run member 7 of the current cycle
rm $EXPERIMENT/status/cycles/cycle_012/members/member_07.json
```

If the status and the actual model output have drifted apart (for example a job was killed between the model finishing and the result being recorded), `wrf-ensembly $EXPERIMENT status reconcile` rebuilds the member advancement of a cycle from the forecast files on disk. Pass `--dry-run` first to see what it would change.


# What to read next

The [Installation](./installation.md) page details how to get WRF-Ensembly installed on your system, including dependencies or for development. The [Usage](./usage.md) page details all commands available in WRF-Ensembly and their usage, including on how to setup your first experiment. The [Configuration](./configuration.md) page details how to configure your experiment, including the namelists. The [WRF](./wrf.md) and [ICON](./icon.md) pages describe what is specific to each model. Information about observations is available in the [Observations](./observations.md) page.