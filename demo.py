# demo.py

import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from toll_plaza_env import TollPlazaEnv
from baseline_agent import get_baseline_action
from stable_baselines3 import DQN

# --- Configuration ---
MODEL_PATH = "models/dqn_toll_plaza_final_model.zip"
DATA_FILE = "weekly_traffic_data.csv"
NUM_LANES = 4

# --- Helper Functions ---

def render_state(agent_name, env_state, info):
    """(Optional) Prints a formatted block for the current state of an environment."""
    queues = env_state
    sim_time = info.get('simulation_time_sec', 0)
    revenue = info.get('revenue', 0)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_index = sim_time // (24 * 3600)
    time_in_day = sim_time % (24 * 3600)
    hour = time_in_day // 3600
    minute = (time_in_day % 3600) // 60
    print(f"--- {agent_name} ---")
    print(f"Time: Day {day_index+1} ({days[day_index]}) - {hour:02d}:{minute:02d}")
    for i in range(NUM_LANES):
        cars = queues[i, 0]
        trucks = queues[i, 1]
        print(f"  Lane {i+1}: 🚗 Cars: {cars:<3} | 🚚 Trucks: {trucks:<3}")
    print(f"Total Revenue: ₹{revenue:,.2f}\n")

def plot_final_results(info_rl, info_baseline):
    """Takes the final info dictionaries and generates comparison graphs."""
    print("\nGenerating final comparison graphs...")
    
    # --- 1. Throughput Graph ---
    rl_throughput = info_rl.get('hourly_throughput', {})
    baseline_throughput = info_baseline.get('hourly_throughput', {})
    
    df_rl = pd.DataFrame(list(rl_throughput.items()), columns=['Hour', 'Throughput'])
    df_rl['Agent'] = 'Smart RL Agent'
    
    df_baseline = pd.DataFrame(list(baseline_throughput.items()), columns=['Hour', 'Throughput'])
    df_baseline['Agent'] = 'Static Baseline'
    
    df_combined = pd.concat([df_rl, df_baseline])
    df_combined['HourOfDay'] = df_combined['Hour'].astype(int) % 24

    plt.figure(figsize=(18, 9))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=df_combined, x='HourOfDay', y='Throughput', hue='Agent', palette='viridis')
    plt.title('Agent vs. Baseline: Average Hourly Throughput', fontsize=20, weight='bold')
    plt.xlabel('Hour of the Day (Averaged across 7 days)', fontsize=14)
    plt.ylabel('Average Vehicles Processed per Hour', fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Agent Type', fontsize=12)
    plt.tight_layout()
    plt.savefig("throughput_comparison.png", dpi=300)
    print("Throughput comparison graph saved to throughput_comparison.png")
    plt.close() # Close the plot to free memory

    # --- 2. Wait Time Graph ---
    rl_wait_time = info_rl.get('avg_wait_time_seconds', 0)
    baseline_wait_time = info_baseline.get('avg_wait_time_seconds', 0)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['Smart RL Agent', 'Static Baseline'], y=[rl_wait_time, baseline_wait_time], palette='plasma')
    plt.title('Average Vehicle Wait Time', fontsize=16, weight='bold')
    plt.ylabel('Average Wait Time (Seconds)', fontsize=12)
    plt.savefig("wait_time_comparison.png", dpi=300)
    print("Wait time comparison graph saved to wait_time_comparison.png")
    plt.close()

# --- Main Demonstration Script ---
if __name__ == "__main__":
    model = DQN.load(MODEL_PATH)
    rl_env = TollPlazaEnv(num_lanes=NUM_LANES, data_filepath=DATA_FILE)
    baseline_env = TollPlazaEnv(num_lanes=NUM_LANES, data_filepath=DATA_FILE)

    obs_rl, info_rl = rl_env.reset()
    obs_baseline, info_baseline = baseline_env.reset()
    
    print("Starting simulation... This will take a moment to run through the full week.")
    terminated = False
    while not terminated:
        # RL Agent
        action_rl, _ = model.predict(obs_rl, deterministic=True)
        obs_rl, _, terminated, _, info_rl = rl_env.step(action_rl)

        # Baseline Agent
        action_baseline = get_baseline_action(obs_baseline)
        obs_baseline, _, _, _, info_baseline = baseline_env.step(action_baseline)

    # --- Print Final Summary ---
    print("\n" + "="*50)
    print("      SIMULATION COMPLETE - FINAL RESULTS")
    print("="*50 + "\n")
    print(f"🧠 SMART RL AGENT:")
    print(f"   - Final Revenue:   ₹{info_rl.get('revenue', 0):,.2f}")
    print(f"   - Avg. Wait Time:  {info_rl.get('avg_wait_time_seconds', 0):.2f} seconds\n")
    
    print(f"📜 STATIC BASELINE:")
    print(f"   - Final Revenue:   ₹{info_baseline.get('revenue', 0):,.2f}")
    print(f"   - Avg. Wait Time:  {info_baseline.get('avg_wait_time_seconds', 0):.2f} seconds")
    
    # --- Generate and Save Graphs ---
    plot_final_results(info_rl, info_baseline)
    
    rl_env.close()
    baseline_env.close()