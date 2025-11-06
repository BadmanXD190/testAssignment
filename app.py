import streamlit as st
import csv
import random
import math
import pandas as pd
from io import BytesIO
from datetime import datetime
from pathlib import Path

# ==============================
# CONFIG: CSV filename to load
# ==============================
CSV_FILENAME = "modified_program_ratings.csv"  # <- load your modified file

# ==============================
# ORIGINAL-STYLE DATA LOADING
# ==============================
@st.cache_data
def read_csv_to_dict(file_path: str):
    """Read CSV into {program: [float,...]}."""
    program_ratings = {}
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]
            program_ratings[program] = ratings
    return program_ratings, header

@st.cache_data
def load_ratings(csv_path=CSV_FILENAME):
    ratings_dict, header = read_csv_to_dict(csv_path)
    return ratings_dict, header

if not Path(CSV_FILENAME).exists():
    st.error(
        f"CSV file '{CSV_FILENAME}' not found. "
        "Please upload it or rename your file to this name."
    )

ratings, header = load_ratings() if Path(CSV_FILENAME).exists() else ({}, [])

# ==============================
# GA PARAMETERS
# ==============================
GEN = 100
POP = 50
CO_R_DEFAULT = 0.8
MUT_R_DEFAULT = 0.02
EL_S = 2

all_programs = list(ratings.keys())
all_time_slots = list(range(6, 24))
HOURS_LABELS = [f"Hour {h}" for h in all_time_slots]

# ==============================
# ORIGINAL FUNCTIONS
# ==============================
def fitness_function(schedule):
    total_rating = 0.0
    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]
    return total_rating

MAX_PERMUTATIONS = 40320  # safety cap

def initialize_pop(programs, time_slots):
    n = len(programs)
    if n == 0:
        return [[]]
    if math.factorial(n) <= MAX_PERMUTATIONS:
        from itertools import permutations
        all_schedules = []
        for perm in permutations(programs, min(n, len(time_slots))):
            all_schedules.append(list(perm))
        return all_schedules
    else:
        remaining = programs[:]
        schedule = []
        for t in range(min(len(time_slots), len(programs))):
            best_prog = max(remaining, key=lambda p: ratings[p][t])
            schedule.append(best_prog)
            remaining.remove(best_prog)
        return [schedule]

def finding_best_schedule(all_schedules):
    best_schedule = []
    max_ratings = -1e18
    for schedule in all_schedules:
        total_ratings = fitness_function(schedule)
        if total_ratings > max_ratings:
            max_ratings = total_ratings
            best_schedule = schedule
    return best_schedule

def crossover(schedule1, schedule2):
    if len(schedule1) < 2:
        return schedule1[:], schedule2[:]
    crossover_point = random.randint(1, len(schedule1) - 1)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

def mutate(schedule):
    if not schedule:
        return schedule
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule

def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP,
                      crossover_rate=CO_R_DEFAULT, mutation_rate=MUT_R_DEFAULT,
                      elitism_size=EL_S):

    population = [initial_schedule[:]]
    for _ in range(population_size - 1):
        s = initial_schedule[:]
        random.shuffle(s)
        population.append(s)

    for _ in range(generations):
        new_population = []
        population.sort(key=lambda sch: fitness_function(sch), reverse=True)
        new_population.extend(population[:elitism_size])

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]
            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)
            new_population.extend([child1, child2])

        population = new_population[:population_size]
    return population[0]

# ==============================
# HELPERS
# ==============================
def build_result_df(schedule):
    rows = []
    for idx, prog in enumerate(schedule):
        hour_label = HOURS_LABELS[idx] if idx < len(HOURS_LABELS) else f"Hour {6+idx}"
        r = ratings[prog][idx]
        rows.append({"Hour": hour_label, "Program": prog, "Rating": round(r, 3)})
    return pd.DataFrame(rows)

def run_pipeline(co_rate, mut_rate, seed=None):
    if seed is not None:
        random.seed(seed)
    all_possible_schedules = initialize_pop(all_programs, all_time_slots)
    initial_best_schedule = finding_best_schedule(all_possible_schedules)
    rem_t_slots = len(all_time_slots) - len(initial_best_schedule)
    ga_schedule = genetic_algorithm(
        initial_best_schedule,
        generations=GEN,
        population_size=POP,
        crossover_rate=co_rate,
        mutation_rate=mut_rate,
        elitism_size=EL_S,
    )
    final_schedule = initial_best_schedule + ga_schedule[:max(0, rem_t_slots)]
    df = build_result_df(final_schedule)
    total = fitness_function(final_schedule)
    return df, total

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

# ==============================
# STREAMLIT UI
# ==============================
st.title("TV Program Scheduling using Genetic Algorithm")

# Page switcher
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["Run Trials", "Saved Results"], index=0)

if "results" not in st.session_state:
    st.session_state.results = {}
if "saved_df" not in st.session_state:
    st.session_state.saved_df = pd.DataFrame(
        columns=["Saved At", "Trial", "Hour", "Program", "Rating", "Crossover Rate", "Mutation Rate", "Total Rating"]
    )

# -----------------------------
# RUN TRIALS PAGE
# -----------------------------
if page == "Run Trials":
    st.sidebar.header("Parameter Settings")
    controls = []
    for i in range(1, 4):
        st.sidebar.subheader(f"Trial {i}")
        co = st.sidebar.slider(f"Crossover Rate (Trial {i})", 0.0, 0.95, CO_R_DEFAULT, 0.01, key=f"co{i}")
        mut = st.sidebar.slider(f"Mutation Rate (Trial {i})", 0.01, 0.05, MUT_R_DEFAULT, 0.01, key=f"mut{i}")
        run_btn = st.sidebar.button(f"Run Trial {i}", key=f"btn{i}")
        controls.append((co, mut, run_btn))
    st.sidebar.markdown("---")
    run_all = st.sidebar.button("Run All 3 Trials")

    def show_results_block(title, co, mut, df, total):
        st.subheader(f"{title} (Crossover={co:.2f}, Mutation={mut:.2f})")
        st.dataframe(df, use_container_width=True)
        st.markdown(f"**Total Rating:** {total:.3f}")
        c1, c2, _ = st.columns([1, 1, 3])
        with c1:
            st.download_button(
                label="Download CSV",
                data=schedule_df_to_csv_bytes(df, title.replace(' Results', ''), co, mut, total),
                file_name=f"{title.replace(' ', '_').lower()}.csv",
                mime="text/csv",
            )
        with c2:
            if st.button("Save Results", key=f"save_{title}"):
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                add_df = df.copy()
                add_df.insert(0, "Trial", title.replace(" Results", ""))
                add_df.insert(0, "Saved At", ts)
                add_df["Crossover Rate"] = round(co, 4)
                add_df["Mutation Rate"] = round(mut, 4)
                add_df["Total Rating"] = round(total, 4)
                st.session_state.saved_df = pd.concat([st.session_state.saved_df, add_df], ignore_index=True)
                st.success("Saved successfully!")

    # Run individual trials
    for i, (co, mut, run_btn) in enumerate(controls, start=1):
        if run_btn and ratings:
            df, total = run_pipeline(co, mut, seed=i)
            st.session_state.results[f"Trial {i}"] = (df, total, co, mut)

    if run_all and ratings:
        for i, (co, mut, _) in enumerate(controls, start=1):
            df, total = run_pipeline(co, mut, seed=i)
            st.session_state.results[f"Trial {i}"] = (df, total, co, mut)

    for trial_name in ["Trial 1", "Trial 2", "Trial 3"]:
        if trial_name in st.session_state.results:
            df, total, co, mut = st.session_state.results[trial_name]
            show_results_block(f"{trial_name} Results", co, mut, df, total)

    if len(st.session_state.results) >= 2:
        st.subheader("Summary of Completed Trials")
        summary = pd.DataFrame([
            {"Trial": t, "Crossover Rate": c, "Mutation Rate": m, "Total Rating": round(total, 3)}
            for t, (df, total, c, m) in st.session_state.results.items()
        ]).sort_values("Trial")
        st.dataframe(summary, use_container_width=True)

# -----------------------------
# SAVED RESULTS PAGE
# -----------------------------
else:
    st.header("Saved Results")
    if st.session_state.saved_df.empty:
        st.info("No results saved yet. Run trials and click Save Results.")
    else:
        st.dataframe(st.session_state.saved_df, use_container_width=True)
        trials_available = sorted(st.session_state.saved_df["Trial"].unique().tolist())
        selected_trials = st.multiselect("Select trial(s)", options=trials_available, default=[])

        st.markdown("""
            <style>
                div.stButton > button {width: 100%; height: 3em;}
            </style>
        """, unsafe_allow_html=True)

        cols = st.columns(4)
        with cols[0]:
            if st.button("Delete Selected"):
                if selected_trials:
                    st.session_state.saved_df = st.session_state.saved_df[
                        ~st.session_state.saved_df["Trial"].isin(selected_trials)
                    ].reset_index(drop=True)
                    st.success(f"Deleted: {', '.join(selected_trials)}")
                    st.rerun()
                else:
                    st.warning("No trials selected.")
        with cols[1]:
            if st.button("Clear All"):
                st.session_state.saved_df = st.session_state.saved_df.iloc[0:0].copy()
                st.success("All results cleared.")
                st.rerun()
        with cols[2]:
            st.download_button(
                "Download All (CSV)",
                data=all_saved_to_csv_bytes(st.session_state.saved_df),
                file_name="all_saved_trials.csv",
                mime="text/csv",
            )
        with cols[3]:
            if selected_trials:
                sel_df = st.session_state.saved_df[
                    st.session_state.saved_df["Trial"].isin(selected_trials)
                ].reset_index(drop=True)
                st.download_button(
                    "Download Selected",
                    data=all_saved_to_csv_bytes(sel_df),
                    file_name="selected_trials.csv",
                    mime="text/csv",
                )
            else:
                st.caption("Select trial(s) to enable download.")
