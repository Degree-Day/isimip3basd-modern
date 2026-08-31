# Third-party notices

## ISIMIP3BASD

This project is based on ISIMIP3BASD 3.0.2, uses its published variable
configuration and thresholds, and includes an xarray/Dask port of its
weighted-sum-preserving MBCnSD statistical-downscaling algorithm.

- Project: ISIMIP3BASD
- Copyright: Potsdam Institute for Climate Impact Research (PIK), 2022
- Release: https://zenodo.org/records/7151476
- DOI: https://doi.org/10.5281/zenodo.7151476
- License: GNU Affero General Public License v3.0 or later

The implementation in this repository uses xarray, xclim, xsdba, Dask, SciPy,
and Zarr. Its MBCnSD numerical core is regression-tested against the archived
release, but the package does not claim bit-for-bit equivalence across all
platforms and interpolation implementations.
