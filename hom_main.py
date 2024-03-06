from tqdm.contrib.concurrent import process_map
from multiprocessing import freeze_support
from hom_parameters import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random


def random_day(p: float) -> int:
    """
    Randomly assign day of phenomenon effect if phenomenon is supposed to occur.
    :param p: probability of phenomenon
    :return:
        - With probability   p (=> phenomenon occurs): a random day in the whole simulation period
        - With probability 1-p (=> phenomenon doesn't occur): NaN
    """
    return random.randint(0, N_DAYS) if random.random() < p else np.nan


def run(iteration=None) -> tuple[list[float], list[float], np.ndarray, np.ndarray]:
    # Day of death/vaccination of each person (NaN if the person doesn't die/get vaccinated)
    death = np.array([random_day(P_DEATH) for _ in range(N_PEOPLE)])
    vaccination = np.array([random_day(P_VACCINATION) for _ in range(N_PEOPLE)])
    # print(np.sum(~np.isnan(vaccination)))

    # Remove vaccinations after death
    vaccination[vaccination > death] = np.nan
    # print(np.sum(~np.isnan(vaccination)))

    # Healthy Vaccine Effect (HVE)
    # p = HVE_P * (1 - d / HVE_K)
    # - p is probability that the vaccine will NOT be given due to HVE
    # - d is difference between the day of death and the day of vaccination (in range 0 to HVE_K)
    # - If the person is vaccinated on his death day (d=0), the probability of HVE is HVE_P
    # - If the person is vaccinated HVE_K days before his death (d=HVE_K), the probability of HVE is 0
    for p, d in zip(np.linspace(HVE_P, 0, HVE_K + 1), range(HVE_K + 1)):
        vaccination[(vaccination + d == death) * np.random.rand(vaccination.shape[0]) > 1 - p] = np.nan

    dead_vacc, dead_norm = [], []
    alive_vacc, alive_norm = [], []

    # Calculate dead and alive for each day
    for i in range(N_DAYS):
        dead_vacc.append(np.sum((death == i) & ~np.isnan(vaccination)))
        dead_norm.append(np.sum((death == i) & np.isnan(vaccination)))
        alive_vacc.append(np.sum(((death > i) | np.isnan(death)) & (vaccination <= i)))
        alive_norm.append(np.sum(((death > i) | np.isnan(death)) & (np.isnan(vaccination) | (vaccination > i))))

    # Calculate day mortality
    mortality_vacc = np.array(dead_vacc) / np.array(alive_vacc) * 365.25
    mortality_norm = np.array(dead_norm) / np.array(alive_norm) * 365.25

    # Convert to weekly mortality
    mortality_vacc = [np.mean(mortality_vacc[i:i + 7]) for i in range(0, N_DAYS, 7)]
    mortality_norm = [np.mean(mortality_norm[i:i + 7]) for i in range(0, N_DAYS, 7)]

    # Histogram
    diff = death - vaccination
    hist_x, hist_y = np.unique(diff[~np.isnan(diff)], return_counts=True)
    # print(np.sum(~np.isnan(vaccination)))

    return mortality_vacc, mortality_norm, hist_x.astype(int), hist_y


if __name__ == "__main__":
    sns.set_theme()
    freeze_support()

    filename = f"{HVE_P=}_{HVE_K=}_{N_RUNS=}"

    # === Run simulations in parallel ===
    # output = process_map(run, range(N_RUNS))
    output = [run() for _ in tqdm(range(N_RUNS))]

    # Extract mortality
    runs_vacc = list(map(lambda m: m[0], output))
    runs_norm = list(map(lambda m: m[1], output))
    runs_hist_x = list(map(lambda m: m[2], output))
    runs_hist_y = list(map(lambda m: m[3], output))

    # === PLOT ACM ===
    x = list(range(len(runs_norm[0])))

    plt.figure(figsize=(11, 6))
    plt.plot(x, np.median(runs_vacc, axis=0), label="Vaccinated - median", color="r")
    plt.plot(x, np.median(runs_norm, axis=0), label="Unvaccinated - median", color="b")
    plt.fill_between(x,
                     np.quantile(runs_vacc, q=0.25, axis=0),
                     np.quantile(runs_vacc, q=0.75, axis=0), color="r", alpha=0.2, label="Vaccinated - 25-75% quantile")
    plt.fill_between(x,
                     np.quantile(runs_norm, q=0.25, axis=0),
                     np.quantile(runs_norm, q=0.75, axis=0), color="b", alpha=0.2, label="Unvaccinated - 25-75% quantile")
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

    plt.plot(big_histogram_x, np.median(big_histogram_y, axis=0), label="Median", color="b")
    plt.fill_between(big_histogram_x,
                     np.quantile(big_histogram_y, q=0.25, axis=0),
                     np.quantile(big_histogram_y, q=0.75, axis=0), color="b", alpha=0.2, label="25-75% quantile")
    plt.xlabel("Number of days since vaccination")
    plt.ylabel("Number of deaths on that day")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"{filename}_hist.svg"))
