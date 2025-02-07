from matplotlib import pyplot as plt
from hom_parameters import *
import seaborn as sns
import numpy as np

sns.set_theme()


def plot_acm(x: list[int], runs_all: list[float], runs_vacc: list[float], runs_norm: list[float]):
    """
    Plot all cause mortality for all, vaccinated and unvaccinated individuals.
    :param x: x of the graph
    :param runs_all: runs with shape of (N_RUNS, ) for all individuals
    :param runs_vacc: runs with shape of (N_RUNS, ) for vaccinated individuals
    :param runs_norm: runs with shape of (N_RUNS, ) for unvaccinated individuals
    :return: None
    """
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
    plt.ylim(bottom=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{RESULT_FILE}_acm.svg")
    plt.clf()


def plot_vaccination_death_difference(runs_hist_x: list[np.ndarray[int]], runs_hist_y: list[np.ndarray[int]]):
    """
    Plot histogram of vaccination-death difference.
    :param runs_hist_x: runs with shape of (N_RUNS, ) for x of the histogram
    :param runs_hist_y: runs with shape of (N_RUNS, ) for y of the histogram
    :return: None
    """
    plt.figure(figsize=(11, 4))
    big_histogram_x = list(range(min(min(i) for i in runs_hist_x), max(max(i) for i in runs_hist_x) + 1))
    big_histogram_y = np.zeros((N_RUNS, len(big_histogram_x)))

    for n, (hx, hy) in enumerate(zip(runs_hist_x, runs_hist_y)):
        for v in big_histogram_x:
            if v in hx:
                big_histogram_y[n, v] = hy[hx == v]

    big_histogram_x = big_histogram_x[:len(big_histogram_y[0]) // 2]
    big_histogram_y = big_histogram_y[:, :len(big_histogram_y[0]) // 2]
    plt.plot(big_histogram_x, np.mean(big_histogram_y, axis=0), label="Mean", color="b")
    plt.fill_between(big_histogram_x,
                     np.mean(big_histogram_y, axis=0) - np.std(big_histogram_y, axis=0),
                     np.mean(big_histogram_y, axis=0) + np.std(big_histogram_y, axis=0), color="b", alpha=0.2,
                     label="Mean ± std")
    plt.ylabel("Number of deaths on that day")
    plt.xlabel("Number of weeks since vaccination")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULT_FILE}_hist.svg")
    plt.clf()


def plot_vaccine_effectiveness(x, runs_effectiveness):
    """
    Plot vaccine effectiveness.
    :param x: x of the graph
    :param runs_effectiveness: runs with shape of (N_RUNS, ) for vaccine effectiveness
    :return: None
    """
    plt.figure(figsize=(11, 4))
    plt.plot(x, np.mean(runs_effectiveness, axis=0), label="Mean", color="g")
    plt.fill_between(x,
                     np.mean(runs_effectiveness, axis=0) - np.std(runs_effectiveness, axis=0),
                     np.mean(runs_effectiveness, axis=0) + np.std(runs_effectiveness, axis=0), color="g", alpha=0.2,
                     label="Mean ± std")
    plt.ylabel("Vaccine effectiveness")
    plt.xlabel("Week number")

    y_min, y_max = plt.ylim()
    y_max_abs = max(abs(y_min), abs(y_max))
    plt.ylim(-y_max_abs, y_max_abs)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULT_FILE}_effectiveness.svg")
    plt.clf()
