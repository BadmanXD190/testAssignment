import streamlit as st
import pandas as pd
import random
from io import BytesIO
from datetime import datetime

# ==============================
# Load and modify dataset
# ==============================
@st.cache_data
def load_data_and_modify():
    df = pd.read_csv("program_ratings.csv")
    df.set_index("Type of Program", inplace=True)

    # Exactly 5 rating edits (document these in your report)
    df.loc["news", "Hour 7"] = 0.2
    df.loc["live_soccer", "Hour 21"] = 0.5
    df.loc["documentary", "Hour 8"] = 0.3
    df.loc["tv_series_a", "Hour 20"] = 0.5
    df.loc["music_program", "Hour 22"] = 0.4
    return df

ratings_df = load_data_and_modify()
programs = list(ratings_df.index)
hours = list(ratings_df.columns)
ratings_matrix = ratings_df.values

# ==============================
# Session initialization
# ==============================
if "results" not in st.session_state:
    st.session_state.results = {}
if "saved_df" not in st.session_state:
    st.session_state.saved_df = pd.DataFrame(
        columns=["Saved At", "Trial", "Hour", "Program", "Rating", "Crossover Rate", "Mutation Rate", "Total Rating"]
    )

# ==============================
# Genetic Algorithm core
# ==============================
def evaluate(schedule):
    return sum(ratings_matrix[prog][hr] for hr, prog in enumerate(schedule))

def per_hour_ratings(schedule):
    return [float(ratings_matrix[prog][hr]) for hr, prog in enumerate(schedule)]

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
    if seed is not None:
        random.seed(seed)
    pop = [[random.randrange(len(programs)) for _ in range(len(hours))] for _ in range(pop_size)]
    fits = [evaluate(ind) for ind in pop]
    best_idx = max(range(pop_size), key=lambda k: fits[k])
    best, best_fit = pop[best_idx].copy(), fits[best_idx]

    for _ in range(generations):
        new_pop = []
        while len(new_pop) < pop_size:
            p1, p2 = select_parent(pop, fits), select_parent(pop, fits)
            if random.random() < co_rate:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            mutate(c1, mut_rate)
            mutate(c2, mut_rate)
            new_pop.extend([c1, c2][: pop_size - len(new_pop)])

        new_fits = [evaluate(ind) for ind in new_pop]
        worst = new_fits.index(min(new_fits))
        new_pop[worst], new_fits[worst] = best.copy(), best_fit
        pop, fits = new_pop, new_fits
        gen_best = max(fits)
        if gen_best > best_fit:
            best_fit = gen_best
            best = pop[fits.index(gen_best)].copy()

    df = pd.DataFrame({
        "Hour": hours,
        "Program": [programs[i] for i in best],
        "Rating": [round(r, 3) for r in per_hour_ratings(best)],
    })
    return df, best_fit

def add_trial_to_saved(trial_name, df, total, co, mut):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    add_df = df.copy()
    add_df.insert(0, "Trial", trial_name)
    add_df.insert(0, "Saved At", ts)
    add_df["Crossover Rate"] = round(co, 4)
    add_df["Mutation Rate"] = round(mut, 4)
    add_df["Total Rating"] = round(total, 4)
    st.session_state.saved_df = pd.concat([st.session_state.saved_df, add_df], ignore_index=True)

# ==============================
# Sidebar Page Navigation
# ==============================
st.sidebar.header("📄 Page")
page = st.sidebar.radio("Go to", ["Run Trials", "Saved Results"], index=0)

# ==============================
# Page: Run Trials
# ==============================
if page == "Run Trials":
    st.title("📺 TV Scheduling using Genetic Algorithm")

    DEFAULT_CO = 0.80
    DEFAULT_MUT = 0.02

    st.sidebar.header("⚙️ Parameter Settings")

    # Trial sliders
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

    def show_results(title, co, mut, df, fitness):
        st.subheader(f"{title} (Crossover Rate = {co:.2f}, Mutation Rate = {mut:.2f})")
        st.dataframe(df, use_container_width=True)
        st.markdown(f"**Total Rating:** {fitness:.3f}")
        cols = st.columns([1, 1, 3])
        with cols[0]:
            if st.button("💾 Save Results", key=f"save_{title}"):
                add_trial_to_saved(title.replace(' Results', ''), df, fitness, co, mut)
                st.success("Result saved successfully!")
        with cols[1]:
            csv = df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download CSV",
                data=csv,
                file_name=f"{title.replace(' ', '_').lower()}.csv",
                mime="text/csv",
            )

    if run_trial1:
        df1, fit1 = run_ga(co_rate1, mut_rate1, seed=1)
        st.session_state.results["Trial 1"] = (df1, fit1, co_rate1, mut_rate1)
    if run_trial2:
        df2, fit2 = run_ga(co_rate2, mut_rate2, seed=2)
        st.session_state.results["Trial 2"] = (df2, fit2, co_rate2, mut_rate2)
    if run_trial3:
        df3, fit3 = run_ga(co_rate3, mut_rate3, seed=3)
        st.session_state.results["Trial 3"] = (df3, fit3, co_rate3, mut_rate3)
    if run_all:
        st.session_state.results = {
            "Trial 1": run_ga(co_rate1, mut_rate1, seed=1),
            "Trial 2": run_ga(co_rate2, mut_rate2, seed=2),
            "Trial 3": run_ga(co_rate3, mut_rate3, seed=3),
        }

    for tname in ["Trial 1", "Trial 2", "Trial 3"]:
        if tname in st.session_state.results:
            df, fit, co, mut = st.session_state.results[tname]
            show_results(f"{tname} Results", co, mut, df, fit)

    if len(st.session_state.results) > 1:
        summary = pd.DataFrame([
            {"Trial": t, "Crossover Rate": c, "Mutation Rate": m, "Total Rating": round(f, 3)}
            for t, (d, f, c, m) in st.session_state.results.items()
        ])
        st.subheader("Summary of Completed Trials")
        st.dataframe(summary, use_container_width=True)

# ==============================
# Page: Saved Results
# ==============================
else:
    st.header("💾 Saved Results")
    if st.session_state.saved_df.empty:
        st.info("No saved results yet. Run trials and click **Save Results**.")
    else:
        st.dataframe(st.session_state.saved_df, use_container_width=True)
        selected = st.multiselect("Select rows to delete", options=st.session_state.saved_df.index)
        cols = st.columns(4)
        with cols[0]:
            if st.button("🗑️ Delete Selected"):
                if selected:
                    st.session_state.saved_df.drop(index=selected, inplace=True)
                    st.session_state.saved_df.reset_index(drop=True, inplace=True)
                    st.success("Selected rows deleted.")
                else:
                    st.warning("No rows selected.")
        with cols[1]:
            if st.button("🧹 Clear All Saved"):
                st.session_state.saved_df = st.session_state.saved_df.iloc[0:0].copy()
                st.success("All saved results cleared.")
        with cols[2]:
            st.download_button(
                "⬇️ Download All as CSV",
                data=st.session_state.saved_df.to_csv(index=False).encode(),
                file_name="saved_results.csv",
                mime="text/csv",
            )
        with cols[3]:
            if selected:
                sel_df = st.session_state.saved_df.loc[selected].reset_index(drop=True)
                st.download_button(
                    "⬇️ Download Selected Rows",
                    data=sel_df.to_csv(index=False).encode(),
                    file_name="selected_saved_results.csv",
                    mime="text/csv",
                )
