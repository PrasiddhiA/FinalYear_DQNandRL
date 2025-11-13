# train.py (REPLACE your current file with this)
import os
from toll_plaza_env import TollPlazaEnv
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import FlattenObservation

# --- Config ---
models_dir = "models"
log_dir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

DATA_FILE = "weekly_traffic_data.csv"
NUM_LANES = 4
N_ENVS = 1       # use 1 for deterministic single-process training; increase later if needed
TIMESTEPS = 2000000  # reduce for dev (e.g., 20000) while debugging

# --- Env factory function that returns a wrapped env ---
def make_wrapped_env():
    # create base env
    base_env = TollPlazaEnv(num_lanes=NUM_LANES, data_filepath=DATA_FILE)
    # Flatten the observation to 1D so SB3's MlpPolicy can consume it directly
    flat_env = FlattenObservation(base_env)
    return flat_env

# Create vectorized environment (make_vec_env accepts a callable)
vec_env = make_vec_env(make_wrapped_env, n_envs=N_ENVS)

print("Environment prepared (flattened + vectorized).")

# --- Define the DQN model ---
model = DQN(
    policy='MlpPolicy',
    env=vec_env,
    verbose=1,
    learning_rate=5e-4,
    buffer_size=50_000,
    learning_starts=1_000,
    batch_size=32,
    gamma=0.99,
    tensorboard_log=log_dir,
    device='auto',
    seed=42
)

# --- Train (you can reduce TIMESTEPS while debugging) ---
print(f"Starting training for {TIMESTEPS:,} timesteps...")
try:
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True, tb_log_name="DQN_Run_1")
finally:
    model_path = os.path.join(models_dir, "dqn_toll_plaza_final_model")
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")
    vec_env.close()
