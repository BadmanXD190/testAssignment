import streamlit as st
import pandas as pd
import random
from io import BytesIO
from datetime import datetime

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
programs = list(ratings_df.index)
hours = list(ratings_df.columns)
ratings_matrix = ratings_df.values

# ==============================
# Session state init
# ==============================
if "results" not in st.session_state:
    st.session_state.results = {}  # "Trial 1" -> (df, total_rating, co, mut)

if "saved_df" not in st.session_state:
    st.session_state.saved_df = pd.DataFrame(
        columns=[
            "Saved At", "Trial", "Hour", "Program", "Rating",
            "Crossover Rate", "Mutation Rate", "Total Rating"
        ]
    )

# ==============================
# GA helpers
# ==============================
def evaluate(schedule):
    return sum(ratings_matrix[prog_idx][hour_idx] for hour_idx, prog_idx in enumerate(schedule))

def per_hour_ratings(schedule):
    return [float(ratings_matrix[prog_idx][hour_idx]) for hour_idx, prog_idx in enumerate(schedule)]

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
            new_pop.extend([c1, c2][:pop_size - len(new_pop)])
        new_fits = [evaluate(ind) for ind in new_pop]
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

def add_trial_to_saved(trial_name: str, df: pd.DataFrame, total_rating: float, co: float, mut: float):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    add_df = df.copy()
    add_df.insert(0, "Trial", trial_name)  # Trial 1 / Trial 2 / Trial 3
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
DEFAULT_MUT = 0.02

if page == "Run Trials":
    # --------------------------
    # Controls
    # --------------------------
    st.sidebar.subheader("Trial 1")
    co_rate1 = st.sidebar.slider("Crossover Rate (Trial 1)", 0.0, 0.95, DEFAULT_CO, 0.01, key="co1")
    mut_rate1 = st.sidebar.slider("Mutation Rate (Trial 1)", 0.01, 0.05, DEFAULT_MUT, 0.01, key="mut1")
    run_trial1 = st.sidebar.button("▶ Run Trial 1", key="btn1")

    st.sidebar.subheader("Trial 2")
    co_rate2 = st.sidebar.slider("Crossover Rate (Trial 2)", 0.0, 0.95, DEFAULT_CO, 0.01, key="co2")
    mut_rate2 = st.sidebar.slider("Mutation Rate (Trial 2)", 0.01, 0.05, DEFAULT_MUT, 0.01, key="mut2")
    run_trial2 = st.sidebar.button("▶ Run Trial 2", key="btn2")

    st.sidebar.subheader("Trial 3")
    co_rate3 = st.sidebar.slider("Crossover Rate (Trial 3)", 0.0, 0.95, DEFAULT_CO, 0.01, key="co3")
    mut_rate3 = st.sidebar.slider("Mutation Rate (Trial 3)", 0.01, 0.05, DEFAULT_MUT, 0.01, key="mut3")
    run_trial3 = st.sidebar.button("▶ Run Trial 3", key="btn3")

    st.sidebar.markdown("---")
    run_all = st.sidebar.button("🚀 Run All 3 Trials", key="btn_all")

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
        st.subheader(f"{title} (CO_R={co:.2f}, MUT_R={mut:.2f})")
        st.dataframe(df, use_container_width=True)
        st.markdown(f"**Total Rating:** {fitness:.3f}")

        cols = st.columns([1, 1, 3])
        with cols[0]:
            st.download_button(
                label="Download Trial CSV",
                data=schedule_df_to_csv_bytes(df, title.replace(' Results', ''), co, mut, fitness),
                file_name=f"{title.replace(' ', '_').lower()}.csv",
                mime="text/csv",
                key=f"dl_{title}",
            )
        with cols[1]:
            if st.button("💾 Save Results", key=f"save_{title}"):
                add_trial_to_saved(title.replace(" Results", ""), df, fitness, co, mut)
                st.success("Saved to 'Saved Results' page")

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

    for trial_name in ["Trial 1", "Trial 2", "Trial 3"]:
        if trial_name in st.session_state.results:
            df, fit, co, mut = st.session_state.results[trial_name]
            show_results(f"{trial_name} Results", co, mut, df, fit)

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
    # Saved Results page (select trials, not rows)
    # ==========================
    st.header("💾 Saved Results")

    if st.session_state.saved_df.empty:
        st.info("No saved results yet. Go to 'Run Trials' and click 'Save Results'.")
    else:
        # Show saved data
        st.dataframe(st.session_state.saved_df, use_container_width=True)

        # Unique trials available in saved results
        trials_available = sorted(st.session_state.saved_df["Trial"].unique().tolist())

        st.markdown("**Select trial(s) to delete or download**")
        selected_trials = st.multiselect(
            "Choose trial(s)",
            options=trials_available,
            default=[],
            help="Operate on whole trials instead of individual rows"
        )

        cols = st.columns([1, 1, 2, 2])
        with cols[0]:
            if st.button("🗑️ Delete Selected Trial(s)"):
                if selected_trials:
                    st.session_state.saved_df = st.session_state.saved_df[
                        ~st.session_state.saved_df["Trial"].isin(selected_trials)
                    ].reset_index(drop=True)
                    st.success(f"Deleted: {', '.join(selected_trials)}")
                else:
                    st.warning("No trials selected.")

        with cols[1]:
            if st.button("🧹 Clear All Saved"):
                st.session_state.saved_df = st.session_state.saved_df.iloc[0:0].copy()
                st.success("All saved results cleared.")

        with cols[2]:
            # Download ALL saved
            st.download_button(
                "⬇️ Download All Saved (CSV)",
                data=all_saved_to_csv_bytes(st.session_state.saved_df),
                file_name="saved_trials_combined.csv",
                mime="text/csv",
            )

        with cols[3]:
            # Download only selected trials
            if selected_trials:
                sel_df = st.session_state.saved_df[
                    st.session_state.saved_df["Trial"].isin(selected_trials)
                ].reset_index(drop=True)
                st.download_button(
                    "⬇️ Download Selected Trial(s)",
                    data=all_saved_to_csv_bytes(sel_df),
                    file_name="saved_trials_selected.csv",
                    mime="text/csv",
                )
            else:
                st.caption("Select trial(s) to enable 'Download Selected Trial(s)'.")
