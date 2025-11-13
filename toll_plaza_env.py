import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import traffic_generator


class TollPlazaEnv(gym.Env):
    """Custom Environment for Toll Plaza Management (Gymnasium-compatible)."""

    metadata = {"render_modes": []}

    def __init__(self, num_lanes=4, data_filepath="weekly_traffic_data.csv"):
        super().__init__()

        # --- Simulation parameters ---
        self.num_lanes = num_lanes
        self.time_per_step = 10  # seconds per simulation step
        self.sim_duration_seconds = 7 * 24 * 3600  # default: one week
        self.data_filepath = data_filepath

        # --- Service times (in seconds) ---
        self.CAR_SERVICE_TIME_MEAN = 12.0
        self.CAR_SERVICE_TIME_STD = 8.0
        self.TRUCK_SERVICE_TIME_MEAN = 30.0
        self.TRUCK_SERVICE_TIME_STD = 11.0

        # --- Tolls (revenue per vehicle) ---
        self.TOLL_PER_CAR = 100
        self.TOLL_PER_TRUCK = 210

        # --- Types ---
        self.LANE_TYPES = {"general": 0, "car_only": 1, "truck_only": 2}
        self.VEHICLE_TYPES = {"car": 0, "truck": 1}

        # --- Action and Observation spaces ---
        num_lane_types = len(self.LANE_TYPES)
        # Each lane can be one of the types, so total combinations = num_lane_types ** num_lanes
        self.action_space = spaces.Discrete(num_lane_types ** self.num_lanes)
        # Observation (FLATTENED): queue lengths for each lane [cars, trucks] -> shape (num_lanes*2,)
        self.observation_space = spaces.Box(
            low=0, high=200, shape=(self.num_lanes * 2,), dtype=np.int32
        )

        # --- Internal states ---
        self.simulation_time = 0
        self.queues = np.zeros((self.num_lanes, 2), dtype=np.int32)
        self.next_event_index = 0
        self.total_revenue = 0.0
        self.total_vehicles_processed = 0
        self.total_wait_time_steps = 0
        self.hourly_throughput = {}

        # --- Load traffic data ---
        print(f"Loading traffic data from {self.data_filepath}...")
        self.arrival_events = traffic_generator.load_traffic_events_from_file(self.data_filepath)
        if self.arrival_events:
            print(f"Loaded {len(self.arrival_events)} vehicle events for the simulation.")
            # assuming events are sorted by time and have key "time"
            self.sim_duration_seconds = self.arrival_events[-1]["time"]
        else:
            print("Warning: No traffic events loaded.")
            self.sim_duration_seconds = 0

    # -------------------------------------------------------------
    # Helper: Decode action integer → lane configuration array
    # -------------------------------------------------------------
    def _map_action_to_config(self, action):
        """
        Converts integer action into lane configuration array.
        Example: for num_lanes=4 → [1, 1, 0, 2]
        """
        config = []
        base = len(self.LANE_TYPES)
        for _ in range(self.num_lanes):
            config.append(action % base)
            action //= base
        return list(reversed(config))

    # -------------------------------------------------------------
    # Reset method
    # -------------------------------------------------------------
    def reset(self, seed=None, options=None):
        """Prepares the environment for a new episode (a new week of simulation)."""
        super().reset(seed=seed)

        self.simulation_time = 0
        self.queues = np.zeros((self.num_lanes, 2), dtype=np.int32)
        self.next_event_index = 0

        self.total_revenue = 0.0
        self.total_wait_time_steps = 0
        self.total_vehicles_processed = 0
        self.hourly_throughput = {}

        # >>> return FLATTENED obs <<<
        obs = self.queues.flatten().astype(np.int32)
        info = {}
        return obs, info

    # -------------------------------------------------------------
    # Step method
    # -------------------------------------------------------------
    def step(self, action):
        """Executes one simulation step based on the current action (lane configuration)."""
        lane_config = self._map_action_to_config(action)

        # If no more events, terminate
        if self.next_event_index >= len(self.arrival_events):
            terminated = True
            truncated = False
            reward = 0.0
            info = {"message": "Simulation finished", "total_revenue": self.total_revenue}
            # >>> return FLATTENED obs <<<
            return self.queues.flatten().astype(np.int32), reward, terminated, truncated, info

        # Consume next arrival event
        event = self.arrival_events[self.next_event_index]
        self.next_event_index += 1

        # --- Safely extract vehicle type ---
        vehicle_type = (
            event.get("VehicleType")
            or event.get("vehicle_type")
            or event.get("Type")
            or event.get("type")
        )
        if vehicle_type is None:
            raise KeyError(f"Missing vehicle type in event: {event}")

        # Assign vehicle to lane (may be None if no valid lane)
        lane_index = self._assign_vehicle_to_lane(vehicle_type, lane_config)
        if lane_index is not None:
            if vehicle_type == "car":
                self.queues[lane_index, 0] += 1
            else:
                self.queues[lane_index, 1] += 1

        # Simple reward: penalize congestion; small positive for revenue
        throughput_penalty = -np.sum(self.queues)
        revenue_inc = self.TOLL_PER_CAR if vehicle_type == "car" else self.TOLL_PER_TRUCK
        self.total_revenue += revenue_inc
        reward = 0.01 * revenue_inc + 0.1 * throughput_penalty

        # Termination when we hit end of arrivals
        terminated = self.next_event_index >= len(self.arrival_events)
        truncated = False

        info = {"total_revenue": self.total_revenue}

        # >>> return FLATTENED obs <<<
        return self.queues.flatten().astype(np.int32), reward, terminated, truncated, info

    # -------------------------------------------------------------
    # Lane operations
    # -------------------------------------------------------------
    def _assign_vehicle_to_lane(self, vehicle_type, lane_config):
        """Assign vehicle to an available lane according to lane type rules."""
        possible_lanes = []
        for i, lane_type in enumerate(lane_config):
            if lane_type == self.LANE_TYPES["general"]:
                possible_lanes.append(i)
            elif lane_type == self.LANE_TYPES["car_only"] and vehicle_type == "car":
                possible_lanes.append(i)
            elif lane_type == self.LANE_TYPES["truck_only"] and vehicle_type == "truck":
                possible_lanes.append(i)
        if not possible_lanes:
            return None
        return random.choice(possible_lanes)

    def _add_vehicle_to_queue(self, lane_index, vehicle_type):
        """Increment queue for the given lane and vehicle type."""
        if vehicle_type == "car":
            self.queues[lane_index, 0] += 1
        else:
            self.queues[lane_index, 1] += 1

    def _process_lane(self, lane_index, lane_type):
        """Simulate service of vehicles in this lane for one time step."""
        served = 0
        for veh_idx, vehicle_kind in enumerate(["car", "truck"]):
            queue_len = self.queues[lane_index, veh_idx]
            if queue_len > 0:
                service_time = self._get_service_time(vehicle_kind)
                if service_time <= self.time_per_step:
                    # Vehicle processed
                    self.queues[lane_index, veh_idx] -= 1
                    served += 1
                    self.total_vehicles_processed += 1
                    self.total_revenue += (
                        self.TOLL_PER_CAR if vehicle_kind == "car" else self.TOLL_PER_TRUCK
                    )
        return served

    def _get_service_time(self, vehicle_type):
        """Sample a random service time for a given vehicle type."""
        if vehicle_type == "car":
            return max(1, np.random.normal(self.CAR_SERVICE_TIME_MEAN, self.CAR_SERVICE_TIME_STD))
        else:
            return max(1, np.random.normal(self.TRUCK_SERVICE_TIME_MEAN, self.TRUCK_SERVICE_TIME_STD))

    # -------------------------------------------------------------
    # Reward computation (not used in current step, kept for future)
    # -------------------------------------------------------------
    def _compute_reward(self, served, arrivals):
        queue_penalty = np.sum(self.queues)
        reward = served * 10.0 - 0.1 * queue_penalty
        return reward

    # -------------------------------------------------------------
    # Render (optional)
    # -------------------------------------------------------------
    def render(self):
        print(
            f"Time {self.simulation_time/3600:.2f}h | Queues: {self.queues.tolist()} | Revenue: {self.total_revenue:.1f}"
        )
