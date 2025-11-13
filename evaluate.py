# evaluate.py
import os
import numpy as np
from stable_baselines3 import DQN
from toll_plaza_env import TollPlazaEnv

# --- Config ---
models_dir = "models"
model_path = os.path.join(models_dir, "dqn_toll_plaza_final_model.zip")


# Load environment and model
env = TollPlazaEnv()
model = DQN.load(model_path, env=env)

episodes = 5
for ep in range(episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

    print(f"Episode {ep + 1}: Total Reward = {total_reward:.2f}, Revenue = {info.get('total_revenue', 0):.2f}")
