import streamlit as st
import pandas as pd
import random

# Load the program ratings data and apply modifications
@st.cache
def load_data():
    """Load the CSV data into a DataFrame and apply rating modifications."""
    df = pd.read_csv('program_ratings.csv')
    df.set_index("Type of Program", inplace=True)
    # Modify 5 rating values based on assumptions:
    df.loc["news", "Hour 7"] = 0.2        # was 0.1, more viewers at 7 AM news
    df.loc["live_soccer", "Hour 21"] = 0.5  # was 0.3, popular match at 9 PM
    df.loc["documentary", "Hour 8"] = 0.3   # was 0.4, fewer viewers for documentary at 8 AM
    df.loc["tv_series_a", "Hour 20"] = 0.5  # was 0.4, strong TV series at 8 PM slot
    df.loc["music_program", "Hour 22"] = 0.4  # was 0.5, slightly lower late-night music show rating
    return df

ratings_df = load_data()
programs = list(ratings_df.index)    # List of program types
hours = list(ratings_df.columns)     # List of hour slots (Hour 6 ... Hour 23)
ratings_matrix = ratings_df.values   # numpy array of ratings for fast access

# GA parameter defaults
DEFAULT_CO = 0.8   # default crossover rate
DEFAULT_MUT = 0.2  # default mutation rate

# Display parameter sliders for three trials
st.title("TV Scheduling with Genetic Algorithm")
st.write("Adjust the Genetic Algorithm parameters for each trial and compare the results.")
st.markdown(f"**Default GA Parameters:** Crossover Rate = {DEFAULT_CO}, Mutation Rate = {DEFAULT_MUT}")

# Trial 1 parameters
st.subheader("Trial 1 Parameters")
co_rate1 = st.slider("Trial 1 – Crossover Rate", 0.0, 0.95, DEFAULT_CO, 0.01)
mut_rate1 = st.slider("Trial 1 – Mutation Rate", 0.01, 0.05, DEFAULT_MUT, 0.01)

# Trial 2 parameters
st.subheader("Trial 2 Parameters")
co_rate2 = st.slider("Trial 2 – Crossover Rate", 0.0, 0.95, DEFAULT_CO, 0.01)
mut_rate2 = st.slider("Trial 2 – Mutation Rate", 0.01, 0.05, DEFAULT_MUT, 0.01)

# Trial 3 parameters
st.subheader("Trial 3 Parameters")
co_rate3 = st.slider("Trial 3 – Crossover Rate", 0.0, 0.95, DEFAULT_CO, 0.01)
mut_rate3 = st.slider("Trial 3 – Mutation Rate", 0.01, 0.05, DEFAULT_MUT, 0.01)

# Genetic Algorithm implementation
def evaluate(schedule):
    """Calculate the total rating for a given schedule (list of program indices by hour)."""
    total_rating = 0.0
    for hour_idx, program_idx in enumerate(schedule):
        total_rating += ratings_matrix[program_idx][hour_idx]
    return total_rating

def select_parent(population, fitnesses):
    """Select one parent index using a tournament of size 2 (pick the fitter of two random individuals)."""
    i, j = random.sample(range(len(population)), 2)
    return population[i] if fitnesses[i] >= fitnesses[j] else population[j]

def crossover(parent1, parent2):
    """One-point crossover: swap suffixes of two parents at a random cut point."""
    if len(parent1) <= 1:
        return parent1.copy(), parent2.copy()  # no crossover possible for length 1
    cut = random.randint(1, len(parent1) - 1)
    child1 = parent1[:cut] + parent2[cut:]
    child2 = parent2[:cut] + parent1[cut:]
    return child1, child2

def mutate(individual, mutation_rate):
    """Mutate an individual schedule by randomly changing each slot with a given probability."""
    for hour_idx in range(len(individual)):
        if random.random() < mutation_rate:
            # Assign a random program index to this timeslot
            individual[hour_idx] = random.randrange(len(programs))

def run_genetic_algorithm(co_rate, mut_rate, generations=100, pop_size=50):
    """Run the GA with given crossover and mutation rates, return the best schedule and its total rating."""
    # Initialize population with random schedules
    population = []
    for _ in range(pop_size):
        # Create a random schedule (random program index for each of the 18 hours)
        individual = [random.randrange(len(programs)) for _ in range(len(hours))]
        population.append(individual)
    # Evaluate initial population fitness
    fitnesses = [evaluate(ind) for ind in population]
    best_individual = population[fitnesses.index(max(fitnesses))].copy()
    best_fitness = max(fitnesses)
    
    # GA main loop
    for gen in range(generations):
        new_population = []
        # Create new_population via selection, crossover, and mutation
        while len(new_population) < pop_size:
            # Select two parents
            parent1 = select_parent(population, fitnesses)
            parent2 = select_parent(population, fitnesses)
            # Crossover to produce children (or copy parents without crossover)
            if random.random() < co_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            # Mutate the offspring
            mutate(child1, mut_rate)
            mutate(child2, mut_rate)
            # Add to new population (ensure we don't exceed pop_size)
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
        # Optional: Elitism - carry over the best individual from previous generation
        current_fitnesses = [evaluate(ind) for ind in new_population]
        worst_idx = current_fitnesses.index(min(current_fitnesses))
        new_population[worst_idx] = best_individual.copy()
        # Update population and fitnesses for next generation
        population = new_population
        fitnesses = [evaluate(ind) for ind in population]
        # Update global best if a better individual is found
        gen_best = max(fitnesses)
        if gen_best > best_fitness:
            best_fitness = gen_best
            best_individual = population[fitnesses.index(gen_best)].copy()
    # Convert best individual (list of program indices) to a DataFrame schedule
    best_schedule = [programs[idx] for idx in best_individual]
    schedule_df = pd.DataFrame({
        "Hour": [h for h in hours], 
        "Program": best_schedule
    })
    return schedule_df, best_fitness

# Run the GA when the user clicks the button
if st.button("Run Genetic Algorithm for 3 Trials"):
    # Perform three trials with the specified parameters
    result1, fitness1 = run_genetic_algorithm(co_rate1, mut_rate1)
    result2, fitness2 = run_genetic_algorithm(co_rate2, mut_rate2)
    result3, fitness3 = run_genetic_algorithm(co_rate3, mut_rate3)
    
    # Display the schedules for each trial
    st.write("## Trial 1 Results (CO_R = %.2f, MUT_R = %.2f)" % (co_rate1, mut_rate1))
    st.table(result1)
    st.write(f"**Total Rating:** {fitness1:.2f}")
    
    st.write("## Trial 2 Results (CO_R = %.2f, MUT_R = %.2f)" % (co_rate2, mut_rate2))
    st.table(result2)
    st.write(f"**Total Rating:** {fitness2:.2f}")
    
    st.write("## Trial 3 Results (CO_R = %.2f, MUT_R = %.2f)" % (co_rate3, mut_rate3))
    st.table(result3)
    st.write(f"**Total Rating:** {fitness3:.2f}")
    
    # Summary of parameters used in each trial
    st.write("## Summary of Trial Parameters")
    summary_df = pd.DataFrame({
        "Trial": [1, 2, 3],
        "Crossover Rate": [co_rate1, co_rate2, co_rate3],
        "Mutation Rate": [mut_rate1, mut_rate2, mut_rate3],
        "Total Rating": [round(fitness1, 2), round(fitness2, 2), round(fitness3, 2)]
    })
    st.table(summary_df)
