# dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from io import BytesIO
from Optimization import spider_monkey
from tensorflow.keras.models import load_model
from aws_scaling import update_aws_scaling_capacity

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Smart Cloud Resource Allocation",
    layout="wide",
    page_icon="☁️"
)

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
if "runtime_result_df" not in st.session_state:
    st.session_state["runtime_result_df"] = None

if "runtime_recommended_vms" not in st.session_state:
    st.session_state["runtime_recommended_vms"] = None

# ---------------------------------------------------
# BASE DIRECTORY
# ---------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------
@st.cache_data
def load_data():
    # Rename your dataset file to this:
    # cloud_workload_trace_2024.csv

    csv_path = os.path.join(base_dir, "Data", "cloud_workload_trace_2024.csv")

    if not os.path.exists(csv_path):
        st.error("Dataset file not found in Data folder.")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # Cleaning
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()
    df = df.ffill()

    required_cols = [
        "CPU Utilization (%)",
        "Memory Utilization (%)",
        "Number of Active VMs"
    ]

    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            return pd.DataFrame()

    df["CPU Utilization (%)"] = df["CPU Utilization (%)"].clip(0, 100)
    df["Memory Utilization (%)"] = df["Memory Utilization (%)"].clip(0, 100)

    return df


df = load_data()

if df.empty:
    st.stop()

original_df = df.copy()

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown("""
# ☁️ Smart Cloud Resource Allocation
### Intelligent VM Allocation for Dynamic Cloud Environments
""")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("⚙️ Simulation Controls")

pop_size = st.sidebar.slider("Spider Monkey Population", 5, 100, 10)
iterations = st.sidebar.slider("Iterations", 10, 200, 30)
rl_method = st.sidebar.radio("Select RL Technique", ["Q-learning", "DQN"])

st.sidebar.markdown("---")
st.sidebar.subheader("🔥 Stress Test Simulation")

cpu_spike = st.sidebar.slider("CPU Spike (%)", 0, 100, 0)
mem_boost = st.sidebar.slider("Memory Boost Multiplier", 1.0, 2.0, 1.0, step=0.1)

# Apply stress
df["CPU Utilization (%)"] = np.clip(df["CPU Utilization (%)"] + cpu_spike, 0, 100)
df["Memory Utilization (%)"] = np.clip(df["Memory Utilization (%)"] * mem_boost, 0, 100)

# ---------------------------------------------------
# LOAD RL MODELS
# ---------------------------------------------------
q_learning_path = os.path.join(base_dir, "RL_Agent", "q_table.pkl")
dqn_model_path = os.path.join(base_dir, "RL_Agent", "models", "dqn_model.h5")

q_table = {}
model = None

if rl_method == "Q-learning":
    if os.path.exists(q_learning_path):
        with open(q_learning_path, "rb") as f:
            q_table = pickle.load(f)

elif rl_method == "DQN":
    if os.path.exists(dqn_model_path):
        model = load_model(dqn_model_path)
        st.sidebar.success("DQN Model Loaded")
    else:
        st.sidebar.warning("DQN Model Not Found")

# ---------------------------------------------------
# RUNTIME TESTER
# ---------------------------------------------------
st.markdown("---")
st.subheader("🧪 Runtime Scenario Tester")

scenario = st.selectbox(
    "Select Workload Scenario",
    ["Custom", "CPU Spike", "Memory Bottleneck", "Underutilized Cluster"]
)

cpu_default = 60
mem_default = 50
active_default = 10

if scenario == "CPU Spike":
    cpu_default = 90
elif scenario == "Memory Bottleneck":
    mem_default = 90
elif scenario == "Underutilized Cluster":
    cpu_default, mem_default, active_default = 20, 25, 30

col1, col2 = st.columns(2)

with col1:
    cpu_input = st.slider("CPU Utilization (%)", 0, 100, cpu_default)
    mem_input = st.slider("Memory Utilization (%)", 0, 100, mem_default)

with col2:
    active_vms_input = st.slider("Current Active VMs", 1, 100, active_default)

if st.button("🚀 Run Runtime Allocation"):

    base_row = df.iloc[-1].copy()

    base_row["CPU Utilization (%)"] = cpu_input
    base_row["Memory Utilization (%)"] = mem_input
    base_row["Number of Active VMs"] = active_vms_input

    runtime_df = pd.DataFrame([base_row] * 10)

    runtime_result_df, _ = spider_monkey.run_optimization(runtime_df, q_table, 5, 5)

    st.session_state["runtime_result_df"] = runtime_result_df
    st.session_state["runtime_recommended_vms"] = int(
        runtime_result_df["Optimized VM Count"].iloc[-1]
    )

# ---------------------------------------------------
# RUNTIME RESULT
# ---------------------------------------------------
if st.session_state["runtime_result_df"] is not None:

    st.success("Optimization Completed")

    st.metric(
        label="Recommended VM Count",
        value=st.session_state["runtime_recommended_vms"]
    )

    if st.button("☁️ Simulate AWS Auto Scaling"):
        update_aws_scaling_capacity(
            st.session_state["runtime_recommended_vms"]
        )

    st.subheader("📈 Runtime VM Allocation Trend")
    st.line_chart(
        st.session_state["runtime_result_df"]["Optimized VM Count"]
    )

# ---------------------------------------------------
# OFFLINE OPTIMIZATION
# ---------------------------------------------------
st.markdown("---")
st.subheader("📊 Historical Workload Optimization")

safe_df = df

result_df, fitness_history = spider_monkey.run_optimization(
    safe_df, q_table, pop_size, iterations
)
# ---------------------------------------------------
# METRICS
# ---------------------------------------------------
m1, m2, m3 = st.columns(3)

m1.metric("Minimum Penalty", round(min(fitness_history), 2))
m2.metric("Maximum Penalty", round(max(fitness_history), 2))
m3.metric(
    "Final VM Recommendation",
    int(result_df["Optimized VM Count"].iloc[-1])
)

# -------------------------------
# SUMMARY CALCULATIONS
# -------------------------------
avg_vm = round(result_df["Optimized VM Count"].mean(), 2)
avg_cost = round(result_df["Cost"].mean(), 2)
avg_energy = round(result_df["Energy"].mean(), 2)
# ---------------------------------------------------
# 🔄 BEFORE vs AFTER COMPARISON
# ---------------------------------------------------
st.markdown("---")
st.markdown("## 🔄 Before vs After Optimization")

before_vm = original_df["Number of Active VMs"].mean()
after_vm = avg_vm

col1, col2 = st.columns(2)

col1.metric("Before Optimization (Avg VM)", round(before_vm, 2))
col2.metric(
    "After Optimization (Avg VM)",
    after_vm,
    delta=round(after_vm - before_vm, 2)
)
cost_before = before_vm * 0.5
cost_after = avg_cost

reduction = ((cost_before - cost_after) / cost_before) * 100

st.metric("💰 Cost Reduction (%)", round(reduction, 2))
violations = original_df[original_df["CPU Utilization (%)"] > 80]
st.metric("⚠️ SLA Violations", len(violations))

# ---------------------------------------------------
# GRAPHS
# ---------------------------------------------------
st.subheader("📌 Optimized VM Allocation Over Iterations")
st.line_chart(result_df["Optimized VM Count"])

st.subheader("💰 Operational Cost Over Iterations")
st.line_chart(result_df["Cost"])

st.subheader("⚡ Energy Consumption Over Iterations")
st.line_chart(result_df["Energy"])

# ---------------------------------------------------
# Q TABLE
# ---------------------------------------------------
if rl_method == "Q-learning" and q_table:

    st.subheader("🧠 Q-Learning Decision Heatmap")

    matrix = np.array([list(v.values()) for v in q_table.values()])

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(matrix, cmap="YlOrRd", ax=ax)
    ax.set_xlabel("Actions")
    ax.set_ylabel("States")
    st.pyplot(fig)
st.markdown("---")


st.markdown("## 🎮 AI-based Cloud Simulator")

sim_cpu = st.slider("Simulated CPU (%)", 0, 100, 70)
sim_mem = st.slider("Simulated Memory (%)", 0, 100, 60)
sim_vm = st.slider("Current VMs", 1, 50, 10)

sim_df = pd.DataFrame([{
    "CPU Utilization (%)": sim_cpu,
    "Memory Utilization (%)": sim_mem,
    "Number of Active VMs": sim_vm
}] * 5)

# Button-based simulation (BEST PRACTICE)
if st.button("🚀 Run AI Simulation"):
    sim_result, _ = spider_monkey.run_optimization(sim_df, q_table, 5, 5)

    rec = int(sim_result["Optimized VM Count"].iloc[-1])

    st.success(f"🤖 AI Recommended VMs: {rec}")

# ---------------------------------------------------
# 📊 FINAL PERFORMANCE SUMMARY (Attractive Ending)
# ---------------------------------------------------
st.markdown("---")
st.markdown("## 🚀 Final System Insights")

col1, col2, col3 = st.columns(3)

col1.metric("⚙️ Avg VM Allocation", avg_vm, delta="Optimized")
col2.metric("💰 Avg Cost ($)", avg_cost)
col3.metric("⚡ Avg Energy", avg_energy)

st.info("These metrics represent overall system efficiency after optimization.")

# ---------------------------------------------------
# 📉 PERFORMANCE INTERPRETATION (SMART ADDITION)
# ---------------------------------------------------
st.markdown("### 📈 Performance Insight")

if avg_vm < 10:
    st.success("✅ System is efficiently utilizing fewer VMs → Cost optimized")
elif avg_vm < 15:
    st.warning("⚖️ Balanced resource allocation detected")
else:
    st.error("⚠️ High VM usage → Possible over-provisioning")


# ---------------------------------------------------
# ⬇️ EXPORT FILES (IMPROVED UI)
# ---------------------------------------------------
st.markdown("---")
st.markdown("## ⬇️ Export Results")

colA, colB = st.columns(2)

# Dataset download
buf = BytesIO()
original_df.to_csv(buf, index=False)
buf.seek(0)

colA.download_button(
    label="📥 Download Full Workload Dataset",
    data=buf,
    file_name="cloud_workload_trace_2024.csv",
    mime="text/csv"
)

# Results download
buf2 = BytesIO()
result_df.to_csv(buf2, index=False)
buf2.seek(0)

colB.download_button(
    label="📊 Download Optimization Results",
    data=buf2,
    file_name="optimized_resource_allocation.csv",
    mime="text/csv"
)
# ---------------------------------------------------
# DATA PREVIEW
# ---------------------------------------------------
st.markdown("---")
st.subheader("📁 Dataset Preview")

st.dataframe(original_df.head(50), width="stretch")
# ---------------------------------------------------
# 🎉 FINAL FOOTER (PRO LOOK)
# ---------------------------------------------------
st.markdown("---")
st.markdown(
    "<center><h4>Smart Cloud Resource Allocation</h4>"
    "<p>Developed using RL + Spider Monkey Optimization</p></center>",
    unsafe_allow_html=True
)