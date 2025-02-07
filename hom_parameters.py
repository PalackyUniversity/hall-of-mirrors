import random
import os

# === PARAMETERS ===
N_PEOPLE = 2_000_000    # number of individuals in the model
N_DAYS = 2 * 52 * 7     # number of simulation days (approximately 2 years)
N_RUNS = 20             # number of simulations

# Death
DEATH_P = 0.02          # probability of death (=> N_PEOPLE * P_DEATH = approximate number of deaths)
DEATH_DISTRIBUTION = lambda: random.randint(0, N_DAYS)

# Vaccination
VACCINATION_P = 0.8     # number of vaccinated (=> N_PEOPLE * P_VACCINATED = approximate number of vaccinations)
VACCINATION_MEAN = 6*4  # mean of the vaccination week (in weeks)
VACCINATION_STD = 1*4   # standard deviation of the vaccination week (in weeks)
VACCINATION_DISTRIBUTION = lambda: max(0, min(N_DAYS, round(random.gauss(VACCINATION_MEAN, VACCINATION_STD))))

# Healthy Vaccine Effect (HVE)
HVE_K = 3    # How many weeks to look into the future to see if the subject will die
HVE_P = 0.5  # Strength of HVE; probability that a vaccine will NOT be given due to HVE at vacc_day == death_day

# Other
YEAR_DAYS = 365.25

# Plots
MIN_DEATH_COUNT = 1     # minimum number of deaths per day to be included in the plot
RESULT_DIR = "result"
RESULT_FILE = os.path.join(RESULT_DIR, f"{HVE_P=}_{HVE_K=}_{N_RUNS=}")

# === SANITY CHECKS ===
assert 0 < HVE_K < N_DAYS, "HVE_K must be smaller than N_days and greater than 0"
assert 0 <= HVE_P <= 1, "Probability of HVE must be between 0 and 1"
assert 0 < DEATH_P < 1, "Probability of death must be between 0 and 1"
assert 0 < VACCINATION_P < 1, "Probability of vaccination must be between 0 and 1"
assert N_DAYS > 0, f"Number of days must be positive"
assert N_PEOPLE > 0, f"Number of people must be positive"

# Convert weeks to days
HVE_K *= 7
VACCINATION_MEAN *= 7
VACCINATION_STD *= 7

os.makedirs(RESULT_DIR, exist_ok=True)
