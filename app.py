import streamlit as st
import pandas as pd
import random

# ------------------------------
# Load and modify dataset
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('program_ratings.csv')
    df.set_index("Type of Program", inplace=True)

    # Modify exactly 5 rating cells (as per assignment)
    df.loc["news", "Hour 7"] = 0.2          # was 0.1
    df.loc["live_soccer", "Hour 21"] = 0.5  # was 0.3
    df.loc["documentary", "Hour 8"] = 0.3   # was 0.4
    df.loc["tv_series_a", "Hour 20"] = 0.5  # was 0.4
    df.loc["music_program", "Hour 22"] = 0.4  # was 0.5
    return df

ratings_df = load_data()
programs = list(ratings_df.index)
hours = list(ratings_df.columns)
ratings_matrix = ratings_df.values

# ------------------------------
# Streamlit UI Layout
# ------------------------------
st.title("📺 TV Scheduling using Genetic Algorithm")

st.sidebar.header("⚙️ Parameter Settings")

# Defaults
DEFAULT_CO = 0.8
DEFAULT_MUT = 0.02

# Trial sliders in sidebar
st.sidebar.subheader("Trial 1")
co_rate1 = st.sidebar.slider("Crossover Rate (Trial 1)", 0.0, 0.95, DEFAULT_CO, 0.01)
mut_rate1 = st.sidebar.slider("Mutation Rate (Trial 1)", 0.01, 0.05, DEFAULT_MUT, 0.01)
run_trial1 = st.sidebar.button("▶ Run Trial 1")

st.sidebar.subheader("Trial 2")
co_rate2 = st.sidebar.slider("Crossover Rate (Trial 2)", 0.0, 0.95, DEFAULT_CO, 0.01)
mut_rate2 = st.sidebar.slider("Mutation Rate (Trial 2)", 0.01, 0.05, DEFAULT_MUT, 0.01)
run_trial2 = st.sidebar.button("▶ Run Trial 2")

st.sidebar.subheader("Trial 3")
co_rate3 = st.sidebar.slider("Crossover Rate (Trial 3)", 0.0, 0.95, DEFAULT_CO, 0.01)
mut_rate3 = st.sidebar.slider("Mutation Rate (Trial 3)", 0.01, 0.05, DEFAULT_MUT, 0.01)
run_trial3 = st.sidebar.button("▶ Run Trial 3")

st.sidebar.markdown("---")
run_all = st.sidebar.button("🚀 Run All 3 Trials")

# ------------------------------
# GA Functions
# ------------------------------
def evaluate(schedule):
    """Calculate total rating."""
    return sum(ratings_matrix[p][h] for h, p in enumerate(schedule))

def per_hour_ratings(schedule):
    """Return rating per hour."""
    return [float(ratings_matrix[p][h]) for h, p in enumerate(schedule)]

def select_parent(pop, fits):
    i, j = random.sample(range(len(pop)), 2)
    return pop[i] if fits[i] >= fits[j] else pop[j]

def crossover(p1, p2):
    if len(p1) <= 1:
        return p1.copy(), p2.copy()
    cut = random.randint(1, len(p1) - 1)
    return p1[:cut] + p2[cut:], p2[:cut] + p1[cut:]

def mutate(ind, rate):
    for h in range(len(ind)):
        if random.random() < rate:
            ind[h] = random.randrange(len(programs))

def run_ga(co_rate, mut_rate, generations=100, pop_size=50, seed=None):
    if seed:
        random.seed(seed)
    pop = [[random.randrange(len(programs)) for _ in range(len(hours))] for _ in range(pop_size)]
    fits = [evaluate(ind) for ind in pop]
    best = pop[fits.index(max(fits))].copy()
    best_fit = max(fits)

    for _ in range(generations):
        new_pop = []
        while len(new_pop) < pop_size:
            p1 = select_parent(pop, fits)
            p2 = select_parent(pop, fits)
            if random.random() < co_rate:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            mutate(c1, mut_rate)
            mutate(c2, mut_rate)
            new_pop += [c1, c2][:pop_size - len(new_pop)]

        new_fits = [evaluate(ind) for ind in new_pop]
        worst = new_fits.index(min(new_fits))
        new_pop[worst] = best.copy()
        new_fits[worst] = best_fit

        pop, fits = new_pop, new_fits
        gen_best = max(fits)
        if gen_best > best_fit:
            best_fit = gen_best
            best = pop[fits.index(gen_best)].copy()

    best_programs = [programs[i] for i in best]
    hour_ratings = per_hour_ratings(best)
    df = pd.DataFrame({
        "Hour": hours,
        "Program": best_programs,
        "Rating": [round(r, 3) for r in hour_ratings]
    })
    return df, best_fit

# ------------------------------
# Display Results
# ------------------------------
def show_results(title, co, mut, df, fitness):
    st.subheader(f"{title} (CO_R={co:.2f}, MUT_R={mut:.2f})")
    st.dataframe(df, use_container_width=True)
    st.markdown(f"**Total Rating:** {fitness:.3f}")

# Run individual or all
if run_trial1:
    df1, fit1 = run_ga(co_rate1, mut_rate1, seed=1)
    show_results("Trial 1 Results", co_rate1, mut_rate1, df1, fit1)

if run_trial2:
    df2, fit2 = run_ga(co_rate2, mut_rate2, seed=2)
    show_results("Trial 2 Results", co_rate2, mut_rate2, df2, fit2)

if run_trial3:
    df3, fit3 = run_ga(co_rate3, mut_rate3, seed=3)
    show_results("Trial 3 Results", co_rate3, mut_rate3, df3, fit3)

if run_all:
    df1, fit1 = run_ga(co_rate1, mut_rate1, seed=1)
    df2, fit2 = run_ga(co_rate2, mut_rate2, seed=2)
    df3, fit3 = run_ga(co_rate3, mut_rate3, seed=3)

    show_results("Trial 1 Results", co_rate1, mut_rate1, df1, fit1)
    show_results("Trial 2 Results", co_rate2, mut_rate2, df2, fit2)
    show_results("Trial 3 Results", co_rate3, mut_rate3, df3, fit3)

    st.subheader("Summary of All Trials")
    summary = pd.DataFrame({
        "Trial": [1, 2, 3],
        "Crossover Rate": [co_rate1, co_rate2, co_rate3],
        "Mutation Rate": [mut_rate1, mut_rate2, mut_rate3],
        "Total Rating": [round(fit1, 3), round(fit2, 3), round(fit3, 3)]
    })
    st.dataframe(summary, use_container_width=True)
