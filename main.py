import os
import ee
import shutil
import toml
import logging
import sys

import geopandas as gpd
import numpy as np
import ursa.sleuth_prep as sp
import xarray as xr

from pathlib import Path
from sleuth_sklearn.estimator import SLEUTH


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s :: [%(levelname)s] :: %(message)s",
    handlers=[
        logging.FileHandler('sleuth.log', 'w', 'utf-8', False),
        logging.StreamHandler(sys.stdout)])

if __name__ == "__main__":
    file_name = sys.argv[1] if len(sys.argv) >= 2 else ''
    sacc = os.getenv('EE_ACC', '')
    credentials = ee.ServiceAccountCredentials(sacc, '.private-key.json')
    ee.Initialize(credentials)

    with open(file_name, "r") as f:
        config = toml.load(f)
    logging.info(f'Loaded {file_name} config file') 

    path_fua = Path(config["paths"]["fua_dir"])
    path_cache = Path(config["paths"]["cache_dir"])
    path_out = Path(config["paths"]["out_dir"])

    path_out.mkdir(parents=True, exist_ok=True)

    df = gpd.read_file(path_fua / "cities_fua.gpkg")
    df = df.reset_index(drop=True)
    nrows = len(df)

    logging.info('Path fua %s', str(path_fua))
    logging.info('Path cache %s', str(path_cache))
    logging.info('Path out %s', str(path_out))

    seed_sequence = np.random.SeedSequence(config["montecarlo"]["random_seed"])
    seeds = seed_sequence.generate_state(len(df))

    for i in config["calibration"]["indices"]:
        try:
            row = df.loc[i]
            if path_cache.exists():
                shutil.rmtree(path_cache)
            path_cache.mkdir(parents=True, exist_ok=True)
            
            city = row["city"]
            country = row["country"]

            logging.info(f'{i+1}/{nrows} {city} - {country}')

            path_results = path_out / str(i).rjust(3, "0")
            path_results.mkdir(parents=True, exist_ok=True)
            with open(path_results / "name", "w") as f:
                f.write(f"{country}\n{city}")

            path_generated = sp.load_or_prep_rasters(country, city, path_fua, path_cache)
            
            with xr.open_dataset(path_generated, mask_and_scale=False) as ds:
                slope = ds["slope"].values
                excluded = ds["excluded"].values
                roads = ds["roads"].values
                roads_dist = ds["dist"].values
                roads_i = ds["road_i"].values
                roads_j = ds["road_j"].values
                
                wanted_years = [year for year in ds["year"].values if year >= config["calibration"]["start_year"]]
                urban = ds["urban"].sel(year=wanted_years).values

            wanted_years = np.array(wanted_years, dtype=np.int32)
            urban = urban.astype(bool)
            slope = slope.astype(np.int32)
            excluded = excluded.astype(np.int32)
            roads = np.nan_to_num(roads).astype(np.int32)
            roads_dist = np.nan_to_num(roads_dist).astype(np.int32)
            roads_i = np.nan_to_num(roads_i).astype(np.int32)
            roads_j = np.nan_to_num(roads_j).astype(np.int32)

            model = SLEUTH(
                n_iters=config["montecarlo"]["iterations"],
                n_refinement_iters=config["refinement"]["iterations"],
                n_refinement_splits=config["refinement"]["splits"],
                n_refinement_winners=config["refinement"]["winners"],
                coef_range_diffusion=config["coefficients"]["diffusion"],
                coef_range_breed=config["coefficients"]["breed"],
                coef_range_road=config["coefficients"]["road"],
                coef_range_slope=config["coefficients"]["slope"],
                coef_range_spread=config["coefficients"]["spread"],
                grid_slope=slope,
                grid_excluded=excluded,
                grid_roads=roads,
                grid_roads_dist=roads_dist,
                grid_roads_i=roads_i,
                grid_roads_j=roads_j,
                crit_slope=config["misc"]["critical_slope"],
                random_state=seeds[i],
                n_jobs=config["multiprocessing"]["threads"],
                verbose=1,
                log_dir=path_results
            )
            logging.info('Fitting model.')
            model.fit(urban, wanted_years)
            logging.info(f'Saved results at {path_results}')
        except Exception as e:
            logging.error(f'Error while processing {i+1}/{nrows} {city} - {country}')
            logging.error(e)
            continue

logging.info('Finished proccessing.')
