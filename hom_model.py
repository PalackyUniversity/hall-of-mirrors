from hom_parameters import *
from typing import Callable
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


def do_simulation() -> tuple[np.ndarray, np.ndarray]:
    """
    Compute one simulation of the model.
    :return:
     - [0]: vaccination - numpy array[int] with length of N_PEOPLE with days of their death
     - [1]: death - numpy array[int] with length of N_PEOPLE with days of their death
    """
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

    return vaccination, death
