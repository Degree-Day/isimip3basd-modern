# ISIMIP3BASD modern workflow

An xarray-based bias-adjustment workflow for the ten climate variables used by
ISIMIP3BASD. It uses xsdba (the package split from `xclim.sdba`) for bias
adjustment, xclim for unit conversion and physical transforms, Dask for
spatial parallelism, and Zarr for chunked output.

This is not a bit-for-bit reimplementation of ISIMIP3BASD. Use the archived
[ISIMIP3BASD 3.0.2 release](https://zenodo.org/records/7151476) when exact
reproduction of published output is required. This package is intended for
new, scalable analyses with explicit, supported algorithm choices.

## Setup

```bash
conda env create -f environment.yml
conda activate isimip3basd-modern
pip install --no-deps -e .
```

## Convert NetCDF to Zarr

```bash
isimip3basd-modern convert data/tas_obs-hist_coarse_1979-2014.nc \
  stores/tas_obs-hist.zarr --chunks time=-1,lat=2,lon=2
```

## Bias adjust

```bash
isimip3basd-modern adjust \
  --reference stores/tas_obs-hist.zarr \
  --historical stores/tas_sim-hist.zarr \
  --simulation stores/tas_sim-fut.zarr \
  --output stores/tas_sim-fut-qdm.zarr \
  --variable tas --quantiles 50 \
  --chunks time=-1,lat=2,lon=2 --workers 4
```

Nearest-neighbor interpolation is the default used by xsdba's quantile-delta
mapping. Choose `--interpolation linear` explicitly when the selected method
and grouping support it.

DQM defaults to day-of-year grouping, as recommended by xsdba. With Gregorian
data, its historical training period should include at least one leap year;
xsdba 0.7 otherwise has a lazy-array template mismatch between 365 computed
groups and the 366 groups implied by the calendar.

The `time` dimension must be one chunk because quantile adjustment needs each
complete time series. Spatial dimensions should be divided into chunks sized
to fit comfortably in worker memory. `--workers N` starts a local Dask cluster;
omit it to use the current scheduler. For a separately launched cluster, pass
its address with `--scheduler-address tcp://host:8786`.

Outputs use Zarr format 3 by default. Pass `--zarr-format 2` when downstream
software has not yet adopted Zarr 3. Format 2 metadata is consolidated; format
3 uses standard metadata because consolidated metadata is not yet part of the
Zarr 3 specification.

The variable name selects a production preset. `--method`, `--kind`,
`--group`, and `--window` are available only when an intentional override is
needed.

## Variable presets

All presets use a 31-day moving day-of-year window, matching the window used
by the ISIMIP3BASD application example.

| Variable | Adjustment | Physical treatment |
| --- | --- | --- |
| `hurs` | additive QDM | logit transform; 0 to 100%; threshold boundary restoration |
| `pr` | multiplicative QDM | xsdba dry-frequency adaptation at 0.1 mm/day; nonnegative |
| `prsnratio` | additive QDM | logit transform; 0 to 1 |
| `ps` | additive DQM | xclim unit harmonization |
| `rlds` | additive DQM | xclim unit harmonization |
| `rsds` | additive QDM | xclim clearness-index transform; 0 to 1 before conversion back |
| `sfcWind` | multiplicative QDM | nonnegative; 0.01 m/s lower threshold |
| `tas` | additive DQM | xclim temperature-unit harmonization |
| `tasrange` | multiplicative QDM | nonnegative; 0.01 K lower threshold |
| `tasskew` | additive QDM | logit transform; 0 to 1 |

The bounded transforms deterministically move boundary values to the preset
threshold before adjustment and restore future boundary masks afterward. This
avoids non-reproducible random jitter while guaranteeing physical output
bounds. The `rsds` preset replaces the legacy empirical upper-bound
climatology with xclim's physically based extraterrestrial-radiation and
clearness-index conversion.

Precipitation frequency adaptation uses `--random-seed 0` by default, making
its stochastic replacement values reproducible. Set another integer for an
independent realization.

## Validate output

```bash
isimip3basd-modern validate stores/tas_sim-fut-qdm.zarr --variable tas
```

Validation checks finite values and physical constraints. It uses xclim data
flags for relative humidity, precipitation, and wind; bounded checks for ratio
variables; and xclim's clearness index for shortwave radiation. The command
prints a JSON report and exits nonzero when validation fails.

## Attribution and license

This project is inspired by and retains the license lineage of ISIMIP3BASD
3.0.2 from the Potsdam Institute for Climate Impact Research. It is an
independent modernization maintained by Degree Day and is not an official
ISIMIP release. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for
upstream provenance.

Licensed under the GNU Affero General Public License v3.0 or later.
