from hom_visualization import plot_acm, plot_vaccination_death_difference, plot_vaccine_effectiveness
from hom_model import do_simulation
from hom_parameters import *

from tqdm.contrib.concurrent import process_map
from multiprocessing import freeze_support
from typing import Any
from tqdm import tqdm

import numpy as np


def run(iteration=None) -> dict[str, Any]:
    vaccination, death = do_simulation()
    dead_vacc  = np.empty((N_DAYS, ), dtype=float)
    dead_norm  = np.empty((N_DAYS, ), dtype=float)
    alive_vacc = np.empty((N_DAYS, ), dtype=float)
    alive_norm = np.empty((N_DAYS, ), dtype=float)

    # Calculate dead and alive for each day
    for i in range(N_DAYS):
        dead_vacc[i] = np.sum((death == i) & ~np.isnan(vaccination))
        dead_norm[i] = np.sum((death == i) & np.isnan(vaccination))
        alive_vacc[i] = np.sum(((death > i) | np.isnan(death)) & (vaccination <= i))
        alive_norm[i] = np.sum(((death > i) | np.isnan(death)) & (np.isnan(vaccination) | (vaccination > i)))

    # Convert to numpy arrays
    dead_vacc = dead_vacc.reshape(-1, 7).mean(axis=1)
    dead_norm = dead_norm.reshape(-1, 7).mean(axis=1)
    alive_vacc = alive_vacc.reshape(-1, 7).mean(axis=1)
    alive_norm = alive_norm.reshape(-1, 7).mean(axis=1)

    # Calculate mortality for both categories
    mortality_all = (dead_vacc + dead_norm) / (alive_vacc + alive_norm) * YEAR_DAYS

    # Remove values where there are not enough data
    dead_vacc[dead_vacc < MIN_DEATH_COUNT] = np.nan
    dead_norm[dead_norm < MIN_DEATH_COUNT] = np.nan

    # Calculate day mortality
    mortality_vacc = dead_vacc / alive_vacc * YEAR_DAYS
    mortality_norm = dead_norm / alive_norm * YEAR_DAYS

    # Histogram
    diff = death - vaccination
    diff = diff[~np.isnan(diff)]
    # diff = np.floor(diff / 7)  # Convert to weeks
    hist_x, hist_y = np.unique(diff, return_counts=True)

    return {
        "mortality": {
            "vacc": mortality_vacc,
            "norm": mortality_norm,
            "all": mortality_all
        },
        "histogram": {
            "x": hist_x.astype(int),
            "y": hist_y
        }
    }

if __name__ == "__main__":
    freeze_support()

    # === Run simulations in parallel ===
    # output = process_map(run, range(N_RUNS))  # Parallel
    output = [run() for _ in tqdm(range(N_RUNS))]  # Single core

    # Extract data
    runs_all = list(map(lambda m: m["mortality"]["all"], output))
    runs_vacc = list(map(lambda m: m["mortality"]["vacc"], output))
    runs_norm = list(map(lambda m: m["mortality"]["norm"], output))
    runs_hist_x = list(map(lambda m: m["histogram"]["x"], output))
    runs_hist_y = list(map(lambda m: m["histogram"]["y"], output))
    runs_effectiveness = list((np.array(runs_norm) - np.array(runs_vacc)) / np.array(runs_norm))

    x = list(range(1, len(runs_norm[0]) + 1))

    plot_acm(x, runs_all, runs_vacc, runs_norm)
    plot_vaccination_death_difference(runs_hist_x, runs_hist_y)
    plot_vaccine_effectiveness(x, runs_effectiveness)
