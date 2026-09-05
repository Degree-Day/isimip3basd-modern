# ISIMIP3BASD modern workflow

An xarray-based bias-adjustment and statistical-downscaling workflow for the
ten climate variables used by ISIMIP3BASD. It uses xsdba (the package split
from `xclim.sdba`) for bias adjustment, a faithful weighted-sum-preserving
MBCnSD implementation for spatial downscaling, xclim for units and physical
transforms, Dask for parallelism, and Zarr for chunked output.

This is not a bit-for-bit reimplementation of ISIMIP3BASD. Use the archived
[ISIMIP3BASD 3.0.2 release](https://zenodo.org/records/7151476) when exact
reproduction of published output is required. This package is intended for
new, scalable analyses with explicit, supported algorithm choices.

The bias adjustment remains univariate. Inter-variable MBCn copula adjustment
is not applied, matching the ISIMIP3b production setting. MBCnSD is retained
for the separate spatial downscaling stage, where its vector dimensions are
fine-grid cells within one coarse cell rather than different climate variables.

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

## Spatially downscale with MBCnSD

Run this after coarse-grid bias adjustment, using historical observations on
the nested target grid:

```bash
isimip3basd-modern downscale \
  --observations-fine stores/tas_obs-hist-fine.zarr \
  --simulation-coarse stores/tas_sim-fut-qdm.zarr \
  --output stores/tas_sim-fut-mbcnsd.zarr \
  --variable tas --iterations 20 --quantiles 50 \
  --chunks time=-1,lat=16,lon=16 --workers 4
```

The implementation follows the archived ISIMIP3BASD 3.0.2 method:

1. Validate that every fine cell is nested entirely within one coarse cell.
2. Bilinearly broadcast the adjusted simulation to the fine grid, using
   periodic longitude and central-cell fallback where neighbors are missing.
3. Apply the stochastic MBCnSD core independently by variable, coarse cell,
   and calendar month, using 20 seeded rotations by default.
4. Preserve the area-weighted coarse signal during intermediate mappings and
   restore variable-specific physical bounds after the final mapping.

The observation and simulation periods may differ, but both must contain all
calendar months. Spatial chunk boundaries should coincide with coarse-cell
boundaries. `--chunks` spatial sizes describe the fine grid and must be
multiples of the grid's downscaling factors. For example, `lat=10,lon=10`
creates one coarse-cell task for 1 degree to 0.1 degree downscaling. Matching
coarse chunks are derived automatically. The complete time axes and fine-cell
vector within each coarse cell are rechunked as core dimensions.

### Global tiled runs

For production runs over the complete domain covered by the nested reference
stores, use the restartable two-dimensional runner:

```bash
python scripts/run_global_downscale_tiles.py \
  --model ACCESS-CM2 --scenario ssp245 \
  --reference-root /data1/era5ref-europe-full \
  --canonical-root /data1/cmip6_fwi_1deg \
  --tile-lat-degrees 5 --tile-lon-degrees 2 --tile-workers 16
```

Unless explicitly overridden, the output is written below
`/data1/cmip6_downscaled_global/MODEL/SCENARIO/STAGE`, preventing different
models, experiments, or historical/future stages from sharing a store.

The global domain and fine-to-coarse refinement factors are discovered from
the reference stores rather than fixed array indices. Tiles have a
one-coarse-cell interpolation halo in both dimensions, and longitude halos
wrap across 0/360 degrees. Coarse bias adjustment and spatial downscaling are
separate restartable stages. Bias adjustment writes a shared global-coordinate
1-degree Zarr store; spatial workers read haloed tiles from that store and the
global fine reference, crop the halo after MBCnSD, and write disjoint output
regions.

Before each spatial task is submitted, empty tile margins are cropped on whole
coarse-cell boundaries. This retains every active 1-degree parent cell and its
global interpolation halo while avoiding MBCnSD setup and I/O for surrounding
ocean cells. Original tile identifiers remain unchanged so older completion
markers and interrupted runs stay restartable.

The stages may be run independently:

```bash
python scripts/run_europe_downscale_tiles.py \
  --model ACCESS-CM2 --scenario ssp245 --regions west east \
  --stages adjust --adjusted-root /data1/cmip6_bias_adjusted_1deg

python scripts/run_europe_downscale_tiles.py \
  --model ACCESS-CM2 --scenario ssp245 --regions west east \
  --stages spatial --adjusted-root /data1/cmip6_bias_adjusted_1deg \
  --output-root /data1/access_europe_downscale_global_context
```

Existing regional `*_adjusted.zarr` products can seed the shared store with
`--seed-adjusted-from`. Since coarse adjustment is pointwise, these values are
reusable; the runner computes only missing halo cells and records coarse-cell
coverage before allowing the spatial stage to start.

The canonical preprocessor and runner accept all ten primary ISIMIP variables
listed below. The runner default remains the four FWI weather inputs (`tas`,
`hurs`, `pr`, and `sfcWind`); the other variables can be selected once matching
fine and coarse reference stores have been prepared.

A common support mask intersects complete model and fine-reference coverage
for every requested variable before any variable is written. Model temperature
must also remain within 130-377 K. This prevents finite ocean fill values from
becoming 150 K temperature sentinels or zero-valued humidity, precipitation,
and wind cells. The current ERA5-Land reference extends from
about 57 degrees south to 90 degrees north; `global` means the complete
reference-covered domain and does not synthesize an Antarctic reference.

`scripts/run_europe_downscale_tiles.py` uses the same generalized engine while
retaining the existing west/east output presets. Region boundaries do not
limit the spatial inputs: every regional tile reads its context from the shared
global-coordinate stores. Faithful MBCnSD still adjusts the fine cells inside
each 1-degree parent cell as one multivariate vector, so daily fields can show
parent-cell block structure even when regional and processing-tile seams are
handled correctly.

Generate global and regional xclim/ETCCDI-style climatology maps to quantify
that structure before accepting a model run:

```bash
python scripts/plot_global_etccdi_qc.py \
  --root /data0/cmip6_downscaled_global/ACCESS-CM2/ssp245/proj/int16/global \
  --output /data0/cmip6_downscaled_global/ACCESS-CM2/ssp245/proj/qc/etccdi.png \
  --zoom-output /data0/cmip6_downscaled_global/ACCESS-CM2/ssp245/proj/qc/etccdi_europe.png \
  --annual-output /data0/cmip6_downscaled_global/ACCESS-CM2/ssp245/proj/qc/annual_indicators.zarr \
  --workers 12
```

The companion JSON report records robust value ranges and the ratio of median
gradients at inherited 1-degree boundaries to gradients within parent cells.
The calculation is spatially tiled and restartable, retains every annual map,
and publishes the annual cube as compressed scaled-int16 Zarr v3. Ocean cells
remain masked for precipitation reductions. In addition to annual mean
temperature and ETCCDI PRCPTOT, Rx1day, and consecutive-dry-days CDD, the cube
contains xclim CDD65 and HDD65 energy degree-days at a 65-degree-Fahrenheit
base. These two energy indicators are included alongside, but are not members
of, the formal ETCCDI core set.

Pass `--scenario historical` to use the canonical `hist` stores; this defaults
to 1993-2014 so 1993-1994 can initialize analyses reported for 1995-2014.
Use a separate output root from future scenarios.

`downscale` automatically checks daily chronology, input metadata, physical
bounds, exact target-grid coordinates, and approximate coarse-scale
conservation. Its `OUTPUT.qc.json` report includes mean, maximum, RMS, and
normalized coarse-scale aggregation errors. The default random seed is zero,
so results are reproducible across Dask execution order.

Fine observations normally require complete active time series. The
`prsnratio` preset uses a 50% minimum because the official inputs contain
expected missing ratios when precipitation or snowfall is absent; MBCnSD fills
these gaps using the archived sampling and zero-fallback rules. Override this
only with an explicit `--min-valid-fraction`.

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

`adjust` also runs QC automatically. Before training it rejects missing or
duplicate dates, non-daily records, calendar mismatches, short training periods,
partial missing time series, infinities, invalid latitude ranges, incompatible
grids, and physical-bound violations. Entirely missing spatial cells are
accepted as a land/ocean mask. The default minimum training record is 10 years;
change it with `--min-training-years`. Use `--min-valid-fraction` to allow a
documented amount of missing data in otherwise active cells.

After writing, the output is reopened and checked for physical bounds, missing
data, and exact preservation of the simulation dimensions and coordinates.
xclim's repeating-value, extreme-value, and climatological-outlier flags are
reported as warnings rather than fatal errors. CF `standard_name` and coordinate
unit issues are warnings; missing variable units are fatal. The complete
machine-readable report is written to `OUTPUT.qc.json` by default.

## Linked variables

A combined dataset can be checked for temperature ordering and consistency,
snowfall bounded by total precipitation, and specific humidity bounded between
zero and one:

```bash
isimip3basd-modern validate-dataset stores/all-adjusted.zarr
```

The same linked inputs can produce analysis-ready derived variables with xarray
and xclim:

```bash
isimip3basd-modern derive stores/all-adjusted.zarr stores/all-derived.zarr
```

When the required inputs are present this adds `tasmin` and `tasmax` from `tas`,
`tasrange`, and `tasskew`; `prsn` from `pr` and `prsnratio`; and `huss` from
`tas`, `hurs`, and `ps` using xclim's specific-humidity routine. Existing
derived variables are retained and checked for consistency by
`validate-dataset`.

For an ISIMIP3b-style sequence, adjust and downscale the ten primary variables,
merge their fine-grid stores, and then run `derive` to produce `tasmin`,
`tasmax`, `prsn`, and `huss`.

### Coastal land cells

For global training, the reference preparation stage can fill LULC-confirmed
land cells absent from ERA5-Land with regular ERA5. ERA5 is bilinearly
interpolated from 0.25 to 0.1 degrees and is used only where ERA5-Land is
missing; valid ERA5-Land values always take precedence. The 1-degree training
reference is then area-aggregated from this composited fine grid:

```bash
python scripts/prepare_era5land_reference.py \
  /data0/data1_archive/era5land-fwi/noon_daily.zarr \
  /data0/era5ref-global-era5fill \
  --era5-daily-root /nas/dat1/ERA5/daily \
  --lulc-land-area /nas/dat1/LULC/global_landarea_30as_km2.tif \
  --variables tas hurs pr sfcWind --workers 12
```

Each variable gets a `source/<variable>.zarr` provenance mask: 0 is outside
mapped land or unavailable, 1 is ERA5-Land, and 2 is the ERA5 fallback. The
regular ERA5 archive contains daily means for temperature, humidity, and wind
and daily totals for precipitation, while the primary ERA5-Land series is
local-noon weather. This semantic difference is recorded in the output
metadata and manifest.

After preparation, render the common four-variable source coverage with:

```bash
python scripts/plot_reference_source_coverage.py \
  /data0/era5ref-global-era5fill \
  reference_source_coverage.png
```

ERA5-Land's center-point support can omit 0.1-degree cells whose footprints
intersect a mapped coastline. After spatial downscaling is complete, add a
conservative one-cell coastal fringe with a common Natural Earth land mask:

```bash
python scripts/fill_global_coastal_cells.py \
  /data1/cmip6_downscaled_global/ACCESS-CM2/ssp245/proj/global \
  --variables tas hurs pr sfcWind --workers 4
```

The restartable stage fills only cells adjacent to existing common support and
intersecting a 10 m Natural Earth land polygon. Each added cell uses its nearest
valid cardinal or diagonal land donor. The saved `coastal_fill_plan.zarr`
ensures every variable receives exactly the same footprint; isolated islands
without an adjacent reference cell remain missing.

## Publication Zarr

Global and regional tiled MBCnSD stores are physically written as scaled
`int16` Zarr v3. Variable-specific scale factors and offsets are applied by
xarray during every region write, so calculations remain floating point while
the data on disk are compact from the first completed tile. The runner refuses
to resume into an older float32 store, preventing mixed physical encodings.

The publication command can rechunk those stores for downstream access without
changing their scaled `int16` representation:

```bash
isimip3basd-modern pack \
  /data1/cmip6_downscaled_global/ACCESS-CM2/ssp245/proj/global/tas_downscaled.zarr \
  /data1/cmip6_published/ACCESS-CM2/ssp245/proj/global/tas.zarr \
  --chunks time=31,lat=256,lon=256 --workers 8
```

For every downscaled store in a global collection:

```bash
python scripts/publish_global_outputs.py \
  /data1/cmip6_downscaled_global/ACCESS-CM2/ssp245/proj \
  /data1/cmip6_published/ACCESS-CM2/ssp245/proj \
  --workers 8 --threads-per-worker 1
```

Tiled and published arrays use a reserved `-32768` missing code, variable-specific
`scale_factor` and `add_offset`, Blosc Zstd level 3 with bitshuffle, and
`time,lat,lon` order. xarray transparently decodes them to floating point.
Publication fails instead of saturating when values exceed the documented
packing range. Every output is reopened and checked for coordinate identity,
maximum round-trip quantization error, infinities, and decoded minimum/maximum;
the report is written beside the store as `STORE.qc.json`.

A 21-year, approximately 10 x 10 degree temperature pilot packed 75 million
values in 6.8 seconds on Sailfish. It reduced 162 MiB of float32 Zarr to 86 MiB
and had 0.002504 K maximum round-trip error.

## Daily fire-weather indices

The Europe workflow preserves the daily Canadian Forest Fire Weather Index
System state variables instead of retaining only a temporal aggregate:

```bash
python scripts/calc_europe_fwi_amax.py \
  --historical-root /data1/access_europe_downscale_historical_global_context \
  --future-root /data1/access_europe_downscale_global_context \
  --out-root /data1/access_europe_downscale_global_context/fwi \
  --workers 8 --threads-per-worker 3
```

Each `daily_fire_weather_indices_PERIOD_HALF.zarr` store contains daily
`ffmc`, `dmc`, `dc`, `isi`, `bui`, and `fwi` as `float32`, chunked by one year
and 10 x 10 spatial cells. Two years before each requested period initialize
the stateful xclim CFFWIS calculation but are not written to the published
daily stores. The mean annual maximum FWI products are then derived by
reopening these stores.

For global production, use the restartable spatially tiled runner and specify
the warm-up and published periods explicitly:

```bash
python scripts/calc_global_fwi.py \
  /data1/cmip6_downscaled_global/ACCESS-CM2/ssp245/proj \
  /data1/cmip6_fwi_global/ACCESS-CM2/ssp245/proj \
  --compute-start 2068-01-01 --compute-end 2090-12-31 \
  --output-start 2070-01-01 --output-end 2090-12-31 \
  --period-label 2070-2090 --tile-size 40 \
  --workers 8 --threads-per-worker 1
```

The runner writes disjoint regions into a shared Zarr store, records one
success marker per spatial tile, rejects incompatible existing stores, and
uses clean CFFWIS metadata rather than inherited temperature attributes. Pack
the completed six-variable store with `isimip3basd-modern pack` for delivery.

## Attribution and license

This project is inspired by and retains the license lineage of ISIMIP3BASD
3.0.2 from the Potsdam Institute for Climate Impact Research. It is an
independent modernization maintained by Degree Day and is not an official
ISIMIP release. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for
upstream provenance.

Licensed under the GNU Affero General Public License v3.0 or later.
