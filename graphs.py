# plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator

# --- Function to extract data from TensorBoard logs ---
def load_tensorboard_logs(log_path):
    # Initialize an EventAccumulator to read the log file
    ea = event_accumulator.EventAccumulator(
        log_path,
        size_guidance={event_accumulator.SCALARS: 0} # 0 means load all scalars
    )
    ea.Reload() # Load the data

    # Get the list of available scalar tags (e.g., 'rollout/ep_rew_mean')
    scalar_tags = ea.Tags()['scalars']
    
    # Extract the ep_rew_mean data
    ep_rew_mean_data = ea.Scalars('rollout/ep_rew_mean')
    
    steps = [e.step for e in ep_rew_mean_data]
    values = [e.value for e in ep_rew_mean_data]

    return pd.DataFrame({'steps': steps, 'reward': values})


# --- Main script ---
if __name__ == "__main__":
    # IMPORTANT: Replace this with the full path to your event file
    log_file_path = "logs\\DQN_Run_1_2\\events.out.tfevents.1760022522.DESKTOP-I65KC5C.25064.0"
    
    # Load the data into a pandas DataFrame
    df = load_tensorboard_logs(log_file_path)

    # Create a professional-looking plot using seaborn
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    plot = sns.lineplot(x='steps', y='reward', data=df, linewidth=2.5)
    
    plot.set_title('Agent Mean Reward Over Training Steps', fontsize=16)
    plot.set_xlabel('Training Timesteps', fontsize=12)
    plot.set_ylabel('Mean Episode Reward', fontsize=12)
    
    # Format the axes for better readability
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    # Save the plot to a file
    plt.savefig("training_reward_graph.png", dpi=300)
    print("Graph saved to training_reward_graph.png")
    plt.show()