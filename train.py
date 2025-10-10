import os
from toll_plaza_env import TollPlazaEnv
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env


# Create directories for saving models and logs
models_dir = "models"
log_dir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

DATA_FILE = "weekly_traffic_data.csv"

env = TollPlazaEnv(num_lanes=4, data_filepath=DATA_FILE)


check_env(env)
print("Environment check passed.")

# 'MlpPolicy' is a standard neural network policy for this type of problem.
# We log data to TensorBoard to visualize the learning process.

print("Defining the DQN model...")
model = DQN(
    policy='MlpPolicy',
    env=env,
    verbose=1,
    learning_rate=0.0005,
    buffer_size=50000,      # Number of past experiences to store for learning
    learning_starts=1000,   # How many steps to take before starting to learn
    batch_size=32,          # Number of experiences to sample for each update
    gamma=0.99,             # Discount factor for future rewards
    tensorboard_log=log_dir
)

# A week-long simulation has ~60,480 steps. Training for 2 million steps
# allows the agent to experience about 33 simulated weeks.

TIMESTEPS = 2000000
print(f"Starting model training for {TIMESTEPS} timesteps...")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
model.learn(
    total_timesteps=TIMESTEPS, 
    progress_bar=True,
    # A unique name for this training run in TensorBoard
    tb_log_name="DQN_Run_1" 
)

print("Training complete. Saving the model...")

model_path = os.path.join(models_dir, "dqn_toll_plaza_final_model")
model.save(model_path)
print(f"Model saved to {model_path}.zip")

env.close()