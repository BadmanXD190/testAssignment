import streamlit as st
import pandas as pd
import random

# ------------------------------
# Data loading with 5 cell edits
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('program_ratings.csv')
    df.set_index("Type of Program", inplace=True)

    # Modify exactly 5 rating cells (document these in your report)
    df.loc["news", "Hour 7"] = 0.2          # was 0.1
    df.loc["live_soccer", "Hour 21"] = 0.5  # was 0.3
    df.loc["documentary", "Hour 8"] = 0.3   # was 0.4
    df.loc["tv_series_a", "Hour 20"] = 0.5  # was 0.4
    df.loc["music_program", "Hour 22"] = 0.4  # was 0.5
    return df

ratings_df = load_data()
programs = list(ratings_df.index)      # program labels
hours = list(ratings_df.columns)       # Hour 6 ... Hour 23
ratings_matrix = ratings_df.values     # fast access for GA

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("TV Scheduling with Genetic Algorithm")

# Drawer (sidebar) controls
st.sidebar.header("Trial Parameters")

# Defaults per your request
DEFAULT_CO = 0.8
DEFAULT_MUT = 0.02  # within [0.01, 0.05]

# Trial 1
st.sidebar.subheader("Trial 1")
co_rate1 = st.sidebar.slider("Crossover Rate (Trial 1)", 0.0, 0.95, DEFAULT_CO, 0.01)
mut_rate1 = st.sidebar.slider("Mutation Rate (Trial 1)", 0.01, 0.2, DEFAULT_MUT, 0.01)

# Trial 2
st.sidebar.subheader("Trial 2")
co_rate2 = st.sidebar.slider("Crossover Rate (Trial 2)", 0.0, 0.95, DEFAULT_CO, 0.01)
mut_rate2 = st.sidebar.slider("Mutation Rate (Trial 2)", 0.01, 0.05, DEFAULT_MUT, 0.01)

# Trial 3
st.sidebar.subheader("Trial 3")
co_rate3 = st.sidebar.slider("Crossover Rate (Trial 3)", 0.0, 0.95, DEFAULT_CO, 0.01)
mut_rate3 = st.sidebar.slider("Mutation Rate (Trial 3)", 0.01, 0.05, DEFAULT_MUT, 0.01)

run_btn = st.sidebar.button("Run Genetic Algorithm for 3 Trials")

# ------------------------------
# Genetic Algorithm
# ------------------------------
def evaluate(schedule):
    """Total rating of a schedule (list of program indices by hour)."""
    return sum(ratings_matrix[prog_idx][hour_idx] for hour_idx, prog_idx in enumerate(schedule))

def per_hour_ratings(schedule):
    """Return list of rating values corresponding to each hour choice."""
    return [float(ratings_matrix[prog_idx][hour_idx]) for hour_idx, prog_idx in enumerate(schedule)]

def select_parent(population, fitnesses):
    """Tournament selection size=2."""
    i, j = random.sample(range(len(population)), 2)
    return population[i] if fitnesses[i] >= fitnesses[j] else population[j]

def crossover(parent1, parent2):
    """One-point crossover."""
    if len(parent1) <= 1:
        return parent1.copy(), parent2.copy()
    cut = random.randint(1, len(parent1) - 1)
    return parent1[:cut] + parent2[cut:], parent2[:cut] + parent1[cut:]

def mutate(individual, mutation_rate):
    """Randomly change slot with probability = mutation_rate."""
    for h in range(len(individual)):
        if random.random() < mutation_rate:
            individual[h] = random.randrange(len(programs))

def run_genetic_algorithm(co_rate, mut_rate, generations=100, pop_size=50, seed=None):
    """Run GA and return schedule DataFrame plus total rating."""
    if seed is not None:
        random.seed(seed)

    # Initialize population
    population = [[random.randrange(len(programs)) for _ in range(len(hours))]
                  for _ in range(pop_size)]
    fitnesses = [evaluate(ind) for ind in population]
    best_idx = max(range(pop_size), key=lambda k: fitnesses[k])
    best_individual = population[best_idx].copy()
    best_fitness = fitnesses[best_idx]

    # Evolution loop
    for _ in range(generations):
        new_population = []
        while len(new_population) < pop_size:
            p1 = select_parent(population, fitnesses)
            p2 = select_parent(population, fitnesses)
            if random.random() < co_rate:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            mutate(c1, mut_rate)
            mutate(c2, mut_rate)
            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)

        # Elitism
        new_fitnesses = [evaluate(ind) for ind in new_population]
        worst_idx = min(range(pop_size), key=lambda k: new_fitnesses[k])
        new_population[worst_idx] = best_individual.copy()
        new_fitnesses[worst_idx] = best_fitness

        population, fitnesses = new_population, new_fitnesses
        gen_best_idx = max(range(pop_size), key=lambda k: fitnesses[k])
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_individual = population[gen_best_idx].copy()

    # Build result table with Hour, Program, Rating columns
    chosen_programs = [programs[idx] for idx in best_individual]
    ratings_each_hour = per_hour_ratings(best_individual)

    schedule_df = pd.DataFrame({
        "Hour": hours,
        "Program": chosen_programs,
        "Rating": [round(r, 3) for r in ratings_each_hour]
    })
    return schedule_df, best_fitness

# ------------------------------
# Run & Display
# ------------------------------
if run_btn:
    # You can fix seeds if you want repeatability per trial
    result1, fitness1 = run_genetic_algorithm(co_rate1, mut_rate1, seed=1)
    result2, fitness2 = run_genetic_algorithm(co_rate2, mut_rate2, seed=2)
    result3, fitness3 = run_genetic_algorithm(co_rate3, mut_rate3, seed=3)

    st.subheader(f"Trial 1 (CO_R={co_rate1:.2f}, MUT_R={mut_rate1:.2f})")
    st.dataframe(result1, use_container_width=True)
    st.markdown(f"**Total Rating:** {fitness1:.3f}")

    st.subheader(f"Trial 2 (CO_R={co_rate2:.2f}, MUT_R={mut_rate2:.2f})")
    st.dataframe(result2, use_container_width=True)
    st.markdown(f"**Total Rating:** {fitness2:.3f}")

    st.subheader(f"Trial 3 (CO_R={co_rate3:.2f}, MUT_R={mut_rate3:.2f})")
    st.dataframe(result3, use_container_width=True)
    st.markdown(f"**Total Rating:** {fitness3:.3f}")

    st.subheader("Summary of Trial Parameters")
    summary_df = pd.DataFrame({
        "Trial": [1, 2, 3],
        "Crossover Rate": [co_rate1, co_rate2, co_rate3],
        "Mutation Rate": [mut_rate1, mut_rate2, mut_rate3],
        "Total Rating": [round(fitness1, 3), round(fitness2, 3), round(fitness3, 3)]
    })
    st.dataframe(summary_df, use_container_width=True)
