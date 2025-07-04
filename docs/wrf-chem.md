# WRF-CHEM

This document provides an overview of the WRF-Chem-specific features of WRF-ensembly.


## Initial conditions

The behaviour of `real.exe` and `wrf.exe` regarding the atm. composition initial condition is controlled by the `chem_in_opt` option in the `&chem` namelist.
If this is set to `1`:

- `real.exe` will attempt to read the initial conditions from `wrf_chem_input` files (not sure exactly what happens) and will do weird things if they are not present.
- `wrf.exe` will read the initial conditions from `wrfinput_d01` files as every other variable.

And if this is set to `0`:
- `real.exe` will not read the initial conditions from `wrf_chem_input` and will write empty fields in `wrfinput_d01`.
- `wrf.exe` will not read initial conditions for the chemistry fields.

Thus, if you are using [interpolator-for-wrfchem](https://github.com/NOA-ReACT/interpolator_for_wrfchem/) to add initial conditions on already prepared `wrfinput_d01` files, you should set `chem_in_opt` to `0` for `real.exe` and `1` for `wrf.exe`. This way, `real.exe` will not try to read the initial conditions from `wrf_chem_input` and will write empty fields in `wrfinput_d01`, while `wrf.exe` will read the initial conditions from `wrfinput_d01` as every other variable.

To automate this behaviour, you can set `manage_chem_in` to `true` in your `config.yml` file (section `data`). This will automatically set `chem_in_opt` to the appropriate value depending on whether you are running `real.exe` or `wrf.exe`. If you set this to `false`, you will have to manually set `chem_in_opt` in your `namelist.input` file.

