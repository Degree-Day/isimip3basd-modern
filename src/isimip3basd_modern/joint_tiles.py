"""Write joint temperature products using one fixed neighborhood per parent."""
import json
from pathlib import Path
from uuid import uuid4

import dask.array as da
import numpy as np
import xarray as xr
import zarr

from . import tiled_runner as runner
from .publication import PACKING_SPECS, PACKED_FILL_VALUE, packing_encoding
from .temperature import COMPONENTS, downscale_temperature
from .temperature_packing import cap_triplet, prepare_packed_triplet

FIELDS = (*COMPONENTS, "tasmin", "tasmax")
METHOD = "joint-kriging-3x3-v2"


def initialize(path, name, parent, fine, factors, identity):
    coords = {"time": parent.time, "lat": fine.lat, "lon": fine.lon}
    if not path.exists():
        shape = tuple(len(c) for c in coords.values())
        data = da.empty(shape, chunks=(len(parent.time), *factors), dtype="float64")
        field = xr.DataArray(data, dims=tuple(coords), coords=coords, name=name, attrs={
            "units": "1" if name == "tasskew" else "K", "spatial_method": METHOD,
            "run_identity": identity, "generation": uuid4().hex,
        })
        field.to_dataset().drop_encoding().to_zarr(
            path, mode="w-", compute=False, zarr_format=3, consolidated=False,
            encoding={name: packing_encoding(name)},
        )
    existing = runner.open_variable(path, name)
    if existing.attrs.get("run_identity") != identity or not all(
        np.array_equal(existing[d], c) for d, c in coords.items()
    ):
        raise ValueError(f"Incompatible output store: {path}")
    return existing.attrs["generation"]


def run_joint_tile(*, tile, reference_root, adjusted_dir, output, region_spec, global_spec,
                   training_period, mask_path, land_path, generation, threads_per_worker):
    runner.configure_worker_runtime(threads_per_worker)
    output = Path(output)
    tile_name = runner._tile_name(tile)
    marker = output / "joint_state" / (tile_name + ".json")
    if marker.exists():
        record = json.loads(marker.read_text())
        if record.get("generation") == generation:
            return record
    mask = runner.open_variable(Path(mask_path), "spatial_valid_mask")
    reference = xr.merge([
        runner.open_variable(Path(reference_root) / "fine" / f"{v}.zarr", v).sel(time=slice(*training_period))
        for v in COMPONENTS], join="exact")
    if land_path:
        land = runner.open_variable(Path(land_path), "lulc_land").sel(lat=reference.lat, lon=reference.lon)
        reference = reference.where(land)
    parent = xr.merge([runner.open_variable(Path(adjusted_dir) / f"{v}.zarr", v) for v in COMPONENTS],
                      join="exact")
    stores = {v: zarr.open_group(output / f"{v}_downscaled.zarr", mode="r+")[v] for v in FIELDS}
    fy, fx = region_spec["lat_factor"], region_spec["lon_factor"]
    counts = {"land": 0, "range_caps": 0}
    for j in range(tile["coarse_lat_start"], tile["coarse_lat_stop"]):
        for i in range(tile["coarse_lon_start"], tile["coarse_lon_stop"]):
            cell = {"coarse_lat_start": j, "coarse_lat_stop": j+1, "coarse_lon_start": i,
                    "coarse_lon_stop": i+1, "fine_lat_start": j*fy, "fine_lat_stop": (j+1)*fy,
                    "fine_lon_start": i*fx, "fine_lon_stop": (i+1)*fx}
            sim, ref, ys, xs = runner.global_tile_contexts(parent, reference, region_spec, global_spec, cell)
            rows, cols = slice(j*fy, (j+1)*fy), slice(i*fx, (i+1)*fx)
            expected = mask.isel(lat=rows, lon=cols).values
            support = np.isfinite(ref.tas.isel(time=0, lat=ys, lon=xs).values)
            if land_path:
                expected = land.sel(lat=mask.lat[rows], lon=mask.lon[cols]).values.astype(bool)
            if not np.array_equal(support, expected):
                raise ValueError(f"Model or reference coverage gap at parent {(j,i)}")
            if not expected.any():
                continue
            fit = downscale_temperature(ref.load(), sim.load()).isel(lat=ys, lon=xs)
            guarded, capped, outside, invalid = cap_triplet(fit)
            if ((outside | invalid) & expected).any():
                raise ValueError(f"Invalid temperature at parent {(j,i)}")
            packed, _ = prepare_packed_triplet(guarded)
            counts["land"] += int(expected.sum())
            counts["range_caps"] += int(capped.sum())
            for v in FIELDS:
                spec = PACKING_SPECS[v]
                a = packed[v].values
                if not np.array_equal(np.isfinite(a), np.broadcast_to(expected, a.shape)):
                    raise ValueError(f"Unexpected output coverage for {v}")
                codes = np.rint((a - spec.add_offset) / spec.scale_factor)
                if np.nanmin(codes) < -32767 or np.nanmax(codes) > 32767:
                    raise ValueError(f"Unrepresentable {v} at {(j,i)}")
                stores[v][:,rows,cols] = np.where(expected, codes, PACKED_FILL_VALUE).astype("int16")
    record = {"tile": tile_name, "generation": generation, **counts}
    marker.parent.mkdir(parents=True, exist_ok=True)
    temp = marker.with_suffix(f".{uuid4().hex}.tmp")
    temp.write_text(json.dumps(record))
    temp.replace(marker)
    return record


def run_region(args, region, spec, global_spec, stage, mask, mask_path):
    output = args.output_root / region
    parent_path = runner.adjusted_store_path(args.adjusted_root,args.model,args.scenario,stage,"tas")
    parent = runner.open_variable(parent_path, "tas")
    # Adjusted stores already carry exactly the selected application period.
    selected = runner.select_simulation_period(parent, args.simulation_start, args.simulation_end)
    if not parent.time.equals(selected.time):
        raise ValueError("Adjusted store and requested simulation periods differ")
    fine = runner._region_fine(runner.open_variable(args.reference_root / "fine/tas.zarr","tas"),spec)
    factors = (spec["lat_factor"], spec["lon_factor"])
    land_path = args.temperature_land_mask or args.reference_root / "lulc_land_mask.zarr"
    if args.temperature_land_mask is not None or land_path.exists():
        world = runner.open_variable(land_path, "lulc_land").sel(lat=fine.lat, lon=fine.lon).values
        if world.dtype != bool or np.any(world & ~mask):
            raise ValueError("Invalid WorldCover mask or unavailable reference/model coverage on land")
        mask = world
    elif global_spec.get("periodic_lon"):
        raise ValueError("Global joint kriging requires --temperature-land-mask or lulc_land_mask.zarr in the reference root")
    else:
        land_path = None
    identity = json.dumps([METHOD, str(args.reference_root.resolve()), str(parent_path.parent.resolve()),
                           list(runner.TRAINING_PERIOD), str(mask_path.resolve()), str(land_path)])
    generations = [initialize(output / f"{v}_downscaled.zarr",v,parent,fine,factors,identity) for v in FIELDS]
    tiles = runner.tile_specs(spec,args.tile_lat_degrees or 5,args.tile_lon_degrees)
    if args.spatial_tile_names:
        requested = set(args.spatial_tile_names)
        if requested - {runner._tile_name(t) for t in tiles}:
            raise ValueError("Unknown joint-kriging tile name")
        tiles = [t for t in tiles if runner._tile_name(t) in requested]
    tasks = [dict(tile=t, reference_root=str(args.reference_root), adjusted_dir=str(parent_path.parent),
                  output=str(output), region_spec=spec, global_spec=global_spec,
                  training_period=runner.TRAINING_PERIOD, mask_path=str(mask_path),
                  land_path=str(land_path) if land_path else None, generation=generations, threads_per_worker=args.threads_per_worker) for t in tiles
             if mask[t["fine_lat_start"]:t["fine_lat_stop"],t["fine_lon_start"]:t["fine_lon_stop"]].any()]
    return runner.run_tile_pool("joint temperature",run_joint_tile,tasks,workers=args.tile_workers)
