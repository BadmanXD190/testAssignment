import streamlit as st
import pandas as pd
import random
from io import BytesIO
from datetime import datetime
import re

# ==============================
# Data loading + exactly 5 edits
# ==============================
@st.cache_data
def load_data_and_modify():
    df = pd.read_csv("program_ratings.csv")
    df.set_index("Type of Program", inplace=True)

    # Exactly 5 rating edits (document these in your report)
    df.loc["news", "Hour 7"] = 0.2           # was 0.1
    df.loc["live_soccer", "Hour 21"] = 0.5   # was 0.3
    df.loc["documentary", "Hour 8"] = 0.3    # was 0.4
    df.loc["tv_series_a", "Hour 20"] = 0.5   # was 0.4
    df.loc["music_program", "Hour 22"] = 0.4 # was 0.5
    return df

ratings_df = load_data_and_modify()
programs = list(ratings_df.index)       # program labels
hours = list(ratings_df.columns)        # Hour 6 ... Hour 23
ratings_matrix = ratings_df.values      # fast lookup

# ==============================
# Session state init
# ==============================
if "results" not in st.session_state:
    # Map: "Trial 1" -> (df, total_rating, co_rate, mut_rate)
    st.session_state.results = {}

if "saved_df" not in st.session_state:
    st.session_state.saved_df = pd.DataFrame(
        columns=[
            "Saved At", "Trial", "Hour", "Program", "Rating",
            "Crossover Rate", "Mutation Rate", "Total Rating"
        ]
    )

# Helper to compute the next saved trial name Trial 1, Trial 2, ...
def next_saved_trial_name():
    if st.session_state.saved_df.empty:
        return "Trial 1"
    trial_col = st.session_state.saved_df["Trial"].dropna().astype(str)
    nums = []
    for t in trial_col:
        m = re.search(r"(\d+)$", t.strip())
        if m:
            nums.append(int(m.group(1)))
    nxt = max(nums) + 1 if nums else 1
    return f"Trial {nxt}"

# ==============================
# Helpers
# ==============================
def evaluate(schedule):
    """Total rating across all hours for a schedule (list of program indices)."""
    return sum(ratings_matrix[prog_idx][hour_idx] for hour_idx, prog_idx in enumerate(schedule))

def per_hour_ratings(schedule):
    """Rating value chosen at each hour for the schedule."""
    return [float(ratings_matrix[prog_idx][hour_idx]) for hour_idx, prog_idx in enumerate(schedule)]

def select_parent(pop, fits):
    """Tournament selection (size 2)."""
    i, j = random.sample(range(len(pop)), 2)
    return pop[i] if fits[i] >= fits[j] else pop[j]

def crossover(p1, p2):
    """One-point crossover."""
    if len(p1) <= 1:
        return p1.copy(), p2.copy()
    cut = random.randint(1, len(p1) - 1)
    return p1[:cut] + p2[cut:], p2[:cut] + p1[:cut] + p2[cut:]

def mutate(ind, rate):
    """Random gene mutation with probability `rate` per hour slot."""
    for h in range(len(ind)):
        if random.random() < rate:
            ind[h] = random.randrange(len(programs))

def run_ga(co_rate, mut_rate, generations=100, pop_size=50, seed=None):
    """Run GA, return (schedule_df, total_rating)."""
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
            # fill to pop_size
            remaining = pop_size - len(new_pop)
            if remaining >= 2:
                new_pop.extend([c1, c2])
            else:
                new_pop.append(c1)

        new_fits = [evaluate(ind) for ind in new_pop]
        # Elitism: keep best solution found so far
        worst_idx = min(range(pop_size), key=lambda k: new_fits[k])
        new_pop[worst_idx] = best.copy()
        new_fits[worst_idx] = best_fit

        pop, fits = new_pop, new_fits
        gen_best_idx = max(range(pop_size), key=lambda k: fits[k])
        if fits[gen_best_idx] > best_fit:
            best_fit = fits[gen_best_idx]
            best = pop[gen_best_idx].copy()

    chosen_programs = [programs[i] for i in best]
    hour_ratings = per_hour_ratings(best)
    schedule_df = pd.DataFrame({
        "Hour": hours,
        "Program": chosen_programs,
        "Rating": [round(r, 3) for r in hour_ratings],
    })
    return schedule_df, best_fit

def schedule_df_to_csv_bytes(df, trial, co, mut, total_rating):
    b = BytesIO()
    out = df.copy()
    out.insert(0, "Trial", trial)
    out["Crossover Rate"] = round(co, 4)
    out["Mutation Rate"] = round(mut, 4)
    out["Total Rating"] = round(total_rating, 4)
    out.to_csv(b, index=False)
    return b.getvalue()

def all_saved_to_csv_bytes(saved_df: pd.DataFrame):
    b = BytesIO()
    saved_df.to_csv(b, index=False)
    return b.getvalue()

def add_trial_to_saved(named_trial: str, df: pd.DataFrame, total_rating: float, co: float, mut: float):
    """Append all rows from a trial's schedule into saved_df with metadata + timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    add_df = df.copy()
    add_df.insert(0, "Trial", named_trial)
    add_df.insert(0, "Saved At", ts)
    add_df["Crossover Rate"] = round(co, 4)
    add_df["Mutation Rate"] = round(mut, 4)
    add_df["Total Rating"] = round(total_rating, 4)
    st.session_state.saved_df = pd.concat([st.session_state.saved_df, add_df], ignore_index=True)

# ==============================
# UI: Sidebar + Page switcher
# ==============================
st.title("📺 TV Scheduling using Genetic Algorithm")

st.sidebar.header("📄 Page")
page = st.sidebar.radio("Go to", ["Run Trials", "Saved Results"], index=0)

st.sidebar.header("⚙️ Parameter Settings")

DEFAULT_CO = 0.80
DEFAULT_MUT = 0.02   # within [0.01, 0.05]

if page == "Run Trials":
    # --------------------------
    # Run Trials controls
    # --------------------------
    # Trial 1 controls
    st.sidebar.subheader("Trial 1")
    co_rate1 = st.sidebar.slider("Crossover Rate (Trial 1)", 0.0, 0.95, DEFAULT_CO, 0.01, key="co1")
    mut_rate1 = st.sidebar.slider("Mutation Rate (Trial 1)", 0.01, 0.05, DEFAULT_MUT, 0.01, key="mut1")
    run_trial1 = st.sidebar.button("▶ Run Trial 1", key="btn1")

    # Trial 2 controls
    st.sidebar.subheader("Trial 2")
    co_rate2 = st.sidebar.slider("Crossover Rate (Trial 2)", 0.0, 0.95, DEFAULT_CO, 0.01, key="co2")
    mut_rate2 = st.sidebar.slider("Mutation Rate (Trial 2)", 0.01, 0.05, DEFAULT_MUT, 0.01, key="mut2")
    run_trial2 = st.sidebar.button("▶ Run Trial 2", key="btn2")

    # Trial 3 controls
    st.sidebar.subheader("Trial 3")
    co_rate3 = st.sidebar.slider("Crossover Rate (Trial 3)", 0.0, 0.95, DEFAULT_CO, 0.01, key="co3")
    mut_rate3 = st.sidebar.slider("Mutation Rate (Trial 3)", 0.01, 0.05, DEFAULT_MUT, 0.01, key="mut3")
    run_trial3 = st.sidebar.button("▶ Run Trial 3", key="btn3")

    st.sidebar.markdown("---")
    run_all = st.sidebar.button("🚀 Run All 3 Trials", key="btn_all")

    # Offer the modified CSV for download (to upload to GitHub for submission)
    def df_to_csv_bytes(df):
        b = BytesIO()
        df.reset_index().to_csv(b, index=False)
        return b.getvalue()

    st.sidebar.download_button(
        "⬇️ Download Modified CSV",
        data=df_to_csv_bytes(ratings_df),
        file_name="program_ratings_modified.csv",
        mime="text/csv",
    )

    # --------------------------
    # Run + persist results
    # --------------------------
    def show_results(title, co, mut, df, fitness):
        # Title uses full words as requested
        st.subheader(f"{title} (Crossover Rate={co:.2f}, Mutation Rate={mut:.2f})")
        st.dataframe(df, use_container_width=True)
        st.markdown(f"**Total Rating:** {fitness:.3f}")

        # Save this trial block
        cols = st.columns([1, 1, 3])
        with cols[0]:
            st.download_button(
                label="Download Trial CSV",
                data=schedule_df_to_csv_bytes(df, title.replace(" Results", ""), co, mut, fitness),
                file_name=f"{title.replace(' ', '_').lower()}.csv",
                mime="text/csv",
                key=f"dl_{title}",
            )
        with cols[1]:
            # Save Results button: assigns sequential Trial N regardless of which run area
            if st.button("Save Results", key=f"save_{title}"):
                trial_name = next_saved_trial_name()
                add_trial_to_saved(trial_name, df, fitness, co, mut)
                st.success(f"Saved as {trial_name} in 'Saved Results'")

    # Button handlers
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
        df1, fit1 = run_ga(co_rate1, mut_rate1, seed=1)
        df2, fit2 = run_ga(co_rate2, mut_rate2, seed=2)
        df3, fit3 = run_ga(co_rate3, mut_rate3, seed=3)
        st.session_state.results = {
            "Trial 1": (df1, fit1, co_rate1, mut_rate1),
            "Trial 2": (df2, fit2, co_rate2, mut_rate2),
            "Trial 3": (df3, fit3, co_rate3, mut_rate3),
        }

    # Display persisted results (no loss on re-run)
    for trial_name in ["Trial 1", "Trial 2", "Trial 3"]:
        if trial_name in st.session_state.results:
            df, fit, co, mut = st.session_state.results[trial_name]
            show_results(f"{trial_name} Results", co, mut, df, fit)

    # Summary appears if ≥ 2 trials are present
    if len(st.session_state.results) >= 2:
        st.subheader("Summary of Completed Trials")
        summary_rows = []
        for t, (df, f, c, m) in st.session_state.results.items():
            summary_rows.append({
                "Trial": t,
                "Crossover Rate": c,
                "Mutation Rate": m,
                "Total Rating": round(f, 3),
            })
        summary_df = pd.DataFrame(summary_rows).sort_values("Trial")
        st.dataframe(summary_df, use_container_width=True)

else:
    # ==========================
    # Saved Results page
    # ==========================
    st.header("💾 Saved Results")

    if st.session_state.saved_df.empty:
        st.info("No saved results yet. Go to 'Run Trials' and click 'Save Results'.")
    else:
        # Show table
        st.dataframe(st.session_state.saved_df, use_container_width=True)

        # Row selection by index
        st.markdown("**Select rows to delete**")
        selected_indices = st.multiselect(
            "Choose row indices",
            options=list(st.session_state.saved_df.index),
            help="You can select one or more rows to delete"
        )

        cols = st.columns([1, 1, 2, 2])
        with cols[0]:
            if st.button("Delete Selected"):
                if len(selected_indices) > 0:
                    st.session_state.saved_df = st.session_state.saved_df.drop(index=selected_indices).reset_index(drop=True)
                    st.success("Selected rows deleted.")
                else:
                    st.warning("No rows selected.")

        with cols[1]:
            if st.button("Clear All Saved"):
                st.session_state.saved_df = st.session_state.saved_df.iloc[0:0].copy()
                st.success("All saved results cleared.")

        with cols[2]:
            st.download_button(
                "Download Saved as CSV",
                data=all_saved_to_csv_bytes(st.session_state.saved_df),
                file_name="saved_trials_combined.csv",
                mime="text/csv",
            )

        with cols[3]:
            # Optional: Download only currently selected rows
            if len(selected_indices) > 0:
                sel_df = st.session_state.saved_df.loc[selected_indices].reset_index(drop=True)
                st.download_button(
                    "Download Selected Rows",
                    data=all_saved_to_csv_bytes(sel_df),
                    file_name="saved_trials_selected.csv",
                    mime="text/csv",
                )
            else:
                st.caption("Select rows to enable 'Download Selected Rows'.")
