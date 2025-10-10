import numpy as np

def get_baseline_action(state):
    """
    Represents a static toll plaza with a fixed lane configuration.
    It ignores the state and always returns the same action.

    Args:
        state (np.array): The current queue lengths (ignored by this agent).

    Returns:
        int: The action ID for our chosen static configuration.
    """
    # To find the action number, we use the formula from the environment:
    # action = 1*(3^0) + 0*(3^1) + 0*(3^2) + 2*(3^3)
    # action = 1*1 + 0*3 + 0*9 + 2*27
    # action = 1 + 0 + 0 + 54 = 55
    #
    # Therefore, the action for this static setup is always 55.
    
    STATIC_LANE_ACTION = 55
    
    return STATIC_LANE_ACTION