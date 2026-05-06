import numpy as np
import pandas as pd
import pickle
import os
import random

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
q_table_path = os.path.join(base_dir, "RL_Agent", "q_table.pkl")

with open(q_table_path, "rb") as f:
    q_table = pickle.load(f)


def fitness(vm_allocation, cpu, mem):
    ideal_util = 60
    if vm_allocation == 0:
        return float('inf')
    return abs(cpu / vm_allocation - ideal_util) + abs(mem / vm_allocation - ideal_util)


def run_optimization(df, q_table, pop_size=10, iterations=30):
    lower_bound, upper_bound = 5, 20
    monkeys = np.random.randint(lower_bound, upper_bound + 1, pop_size)

    best_fitness_history = []
    vm_trend = []
    cost_list = []
    energy_list = []

    #  NEW: logs for CPU & Memory per iteration
    cpu_log = []
    mem_log = []

    n_rows = len(df)

    for iteration in range(iterations):
        # cycle through df rows safely even if df is very small
        cpu = df["CPU Utilization (%)"].iloc[iteration % n_rows]
        mem = df["Memory Utilization (%)"].iloc[iteration % n_rows]

        cpu_log.append(cpu)
        mem_log.append(mem)

        fitness_scores = []
        for vm in monkeys:
            score = fitness(vm, cpu, mem)
            fitness_scores.append(score)

        best_index = np.argmin(fitness_scores)
        best_vm = monkeys[best_index]
        best_penalty = fitness_scores[best_index]

        # Save best values
        vm_trend.append(best_vm)
        best_fitness_history.append(best_penalty)

        # Cost & Energy models
        cost = best_vm * 0.3 + random.uniform(1, 3)
        energy = best_vm * random.uniform(0.5, 1.2)
        cost_list.append(round(cost, 2))
        energy_list.append(round(energy, 2))

        # Update population based on Q-learning or fallback
        new_monkeys = []
        for i, vm in enumerate(monkeys):
            cpu_bin = np.digitize([cpu], np.linspace(0, 100, 6))[0]
            mem_bin = np.digitize([mem], np.linspace(0, 100, 6))[0]
            state = (cpu_bin, mem_bin)

            if state in q_table:
                q_action = max(q_table[state], key=q_table[state].get)
                if q_action == 'Add_VM':
                    vm += 1
                elif q_action == 'Remove_VM':
                    vm -= 1
            else:
                # Exploration if state not in Q-table
                vm = best_vm + random.randint(-2, 2)

            vm = np.clip(vm, lower_bound, upper_bound)
            new_monkeys.append(vm)

        monkeys = np.array(new_monkeys)

        if iteration == iterations - 1:
          print(f"Optimization Completed | Best VM = {best_vm}")

    #  All lists have SAME length = iterations
    result_df = pd.DataFrame({
        "CPU Utilization (%)": cpu_log,
        "Memory Utilization (%)": mem_log,
        "Optimized VM Count": vm_trend,
        "Cost": cost_list,
        "Energy": energy_list
    })

    return result_df, best_fitness_history
