import os

# === PARAMETERS ===
N_PEOPLE = 1_000_000  # number of individuals in the model
N_DAYS = 2 * 52 * 7   # number of simulation days (approximately 2 years)
N_RUNS = 100          # number of simulations

P_DEATH = 0.1         # probability of death (=> N_PEOPLE * P_DEATH = approximate number of deaths)
P_VACCINATION = 0.5   # number of vaccinated (=> N_PEOPLE * P_VACCINATED = approximate number of vaccinations)

# Healthy Vaccine Effect (HVE)
HVE_K = 12    # How many weeks to look into the future to see if the subject will die
HVE_P = 0  # Strength of HVE; probability that a vaccine will NOT be given due to HVE at vacc_day == death_day

RESULT_DIR = "result"

# === SANITY CHECKS ===
assert 0 < HVE_K < N_DAYS, "HVE_K must be smaller than N_days and greater than 0"
assert 0 <= HVE_P <= 1, "Probability of HVE must be between 0 and 1"
assert 0 < P_DEATH < 1, "Probability of death must be between 0 and 1"
assert 0 < P_VACCINATION < 1, "Probability of vaccination must be between 0 and 1"
assert N_DAYS > 0, f"Number of days must be positive"
assert N_PEOPLE > 0, f"Number of people must be positive"

HVE_K *= 7    # Convert weeks to days
os.makedirs(RESULT_DIR, exist_ok=True)
