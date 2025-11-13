import numpy as np

def get_baseline_action(state, num_lanes=4):
    """
    Static toll plaza baseline agent.
    Always returns the same lane configuration:
      Lane 1 -> car only
      Lane 2 -> general
      Lane 3 -> general
      Lane 4 -> truck only
    """
    lane_config = [1, 0, 0, 2]  # matches environment encoding

    # Convert configuration to single action integer (base-3 encoding)
    action = 0
    for i, val in enumerate(lane_config):
        action += val * (3 ** i)

    return action
