from tqdm.contrib.concurrent import process_map
from multiprocessing import freeze_support
from hom_parameters import *
from typing import Callable
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random


def random_day(distribution: Callable[[], int], p: float) -> int:
    """
    Randomly assign day of phenomenon effect if phenomenon is supposed to occur.
    :param p: probability of phenomenon
    :param distribution: distribution to select the day from
    :return:
        - With probability   p (=> phenomenon occurs): a random day in the whole simulation period
        - With probability 1-p (=> phenomenon doesn't occur): NaN
    """
    return distribution() if random.random() < p else np.nan


def run(iteration=None) -> dict:
    # Day of death/vaccination of each person (NaN if the person doesn't die/get vaccinated)
    death = np.array([random_day(DEATH_DISTRIBUTION, DEATH_P) for _ in range(N_PEOPLE)])
    vaccination = np.array([random_day(VACCINATION_DISTRIBUTION, VACCINATION_P) for _ in range(N_PEOPLE)])
    # print(np.sum(~np.isnan(vaccination)))

    # Remove vaccinations after death
    vaccination[vaccination > death] = np.nan
    # print(np.sum(~np.isnan(vaccination)))

    # Remove vaccinations on the day of death with 50% probability, so it is fair
    vaccination[(death == vaccination) & (np.random.rand(len(vaccination)) <= 0.5)] = np.nan

    # Healthy Vaccine Effect (HVE)
    # p = HVE_P * (1 - d / HVE_K)
    # - p is probability that the vaccine will NOT be given due to HVE
    # - d is difference between the day of death and the day of vaccination (in range 0 to HVE_K)
    # - If the person is vaccinated on his death day (d=0), the probability of HVE is HVE_P
    # - If the person is vaccinated HVE_K days before his death (d=HVE_K), the probability of HVE is 0
    for p, d in zip(np.linspace(HVE_P, 0, HVE_K + 1), range(HVE_K + 1)):
        vaccination[(vaccination + d == death) & (np.random.rand(vaccination.shape[0]) <= p)] = np.nan

    dead_vacc, dead_norm = [], []
    alive_vacc, alive_norm = [], []

    # Calculate dead and alive for each day
    for i in range(N_DAYS):
        dead_vacc.append(np.sum((death == i) & ~np.isnan(vaccination)))
        dead_norm.append(np.sum((death == i) & np.isnan(vaccination)))
        alive_vacc.append(np.sum(((death > i) | np.isnan(death)) & (vaccination <= i)))
        alive_norm.append(np.sum(((death > i) | np.isnan(death)) & (np.isnan(vaccination) | (vaccination > i))))

    # Convert to numpy arrays
    dead_vacc, dead_norm = np.array(dead_vacc, dtype=float), np.array(dead_norm, dtype=float)
    alive_vacc, alive_norm = np.array(alive_vacc, dtype=float), np.array(alive_norm, dtype=float)

    # Remove values where there are not enough data
    dead_vacc[dead_vacc < MIN_DEATH_COUNT] = np.nan
    dead_norm[dead_norm < MIN_DEATH_COUNT] = np.nan

    # Calculate day mortality
    mortality_vacc = dead_vacc / alive_vacc * YEAR_DAYS
    mortality_norm = dead_norm / alive_norm * YEAR_DAYS
    mortality_all = (dead_vacc + dead_norm) / (alive_vacc + alive_norm) * YEAR_DAYS

    # Convert to weekly mortality
    mortality_vacc = [np.mean(mortality_vacc[i:i + 7]) for i in range(0, N_DAYS, 7)]
    mortality_norm = [np.mean(mortality_norm[i:i + 7]) for i in range(0, N_DAYS, 7)]
    mortality_all = [np.mean(mortality_all[i:i + 7]) for i in range(0, N_DAYS, 7)]

    # Histogram
    diff = death - vaccination
    diff = np.floor(diff[~np.isnan(diff)] / 7)  # Convert to weeks
    hist_x, hist_y = np.unique(diff, return_counts=True)
    # print(np.sum(~np.isnan(vaccination)))

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
    sns.set_theme()
    freeze_support()

    filename = f"{HVE_P=}_{HVE_K=}_{N_RUNS=}"

    # === Run simulations in parallel ===
    # output = process_map(run, range(N_RUNS))
    output = [run() for _ in tqdm(range(N_RUNS))]

    # Extract mortality
    runs_all = list(map(lambda m: m["mortality"]["all"], output))
    runs_vacc = list(map(lambda m: m["mortality"]["vacc"], output))
    runs_norm = list(map(lambda m: m["mortality"]["norm"], output))
    runs_hist_x = list(map(lambda m: m["histogram"]["x"], output))
    runs_hist_y = list(map(lambda m: m["histogram"]["y"], output))
    runs_effectiveness = list((np.array(runs_norm) - np.array(runs_vacc)) / np.array(runs_norm))

    # === PLOT ACM ===
    x = list(range(1, len(runs_norm[0]) + 1))

    plt.figure(figsize=(11, 4))
    plt.plot(x, np.mean(runs_vacc, axis=0), label="Mean: vaccinated", color="r")
    plt.plot(x, np.mean(runs_norm, axis=0), label="Mean: unvaccinated", color="b")
    plt.plot(x, np.mean(runs_all, axis=0), label="Mean: all", color="k")
    plt.fill_between(x,
                     np.mean(runs_vacc, axis=0) - np.std(runs_vacc, axis=0),
                     np.mean(runs_vacc, axis=0) + np.std(runs_vacc, axis=0), color="r", alpha=0.2,
                     label="Mean ± std: vaccinated")
    plt.fill_between(x,
                     np.mean(runs_norm, axis=0) - np.std(runs_norm, axis=0),
                     np.mean(runs_norm, axis=0) + np.std(runs_norm, axis=0), color="b", alpha=0.2,
                     label="Mean ± std: unvaccinated")
    plt.ylabel("ACM [deaths per person-year]")
    plt.xlabel("Week number")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"{filename}_acm.svg"))
    plt.clf()

    # === PLOT HISTOGRAM OF VACCINATION-DEATH DIFFERENCE ===
    big_histogram_x = list(range(min(min(i) for i in runs_hist_x), max(max(i) for i in runs_hist_x) + 1))
    big_histogram_y = np.zeros((len(output), len(big_histogram_x)))

    for n, (hx, hy) in enumerate(zip(runs_hist_x, runs_hist_y)):
        for v in big_histogram_x:
            if v in hx:
                big_histogram_y[n, v] = hy[hx == v]

    plt.plot(big_histogram_x, np.mean(big_histogram_y, axis=0), label="Mean", color="b")
    plt.fill_between(big_histogram_x,
                     np.mean(big_histogram_y, axis=0) - np.std(big_histogram_y, axis=0),
                     np.mean(big_histogram_y, axis=0) + np.std(big_histogram_y, axis=0), color="b", alpha=0.2,
                     label="Mean ± std")
    plt.ylabel("Number of deaths on that day")
    plt.xlabel("Week number")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"{filename}_hist.svg"))
    plt.clf()

    # === PLOT VACCINE EFFECTIVENESS ===
    plt.plot(x, np.mean(runs_effectiveness, axis=0), label="Mean", color="g")
    plt.fill_between(x,
                     np.mean(runs_effectiveness, axis=0) - np.std(runs_effectiveness, axis=0),
                     np.mean(runs_effectiveness, axis=0) + np.std(runs_effectiveness, axis=0), color="g", alpha=0.2,
                     label="Mean ± std")
    plt.ylabel("Vaccine effectiveness")
    plt.xlabel("Week number")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"{filename}_effectiveness.svg"))
