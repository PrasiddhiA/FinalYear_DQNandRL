import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traffic_generator


class TollPlazaEnv(gym.Env):
    '''Custom Environment for Toll Plaza Management'''

    def __init__(self, num_lanes=4, data_filepath="weekly_traffic_data.csv"):
        super(TollPlazaEnv, self).__init__()

        self.num_lanes = num_lanes
        self.time_per_step = 10
        self.sim_duration_seconds = 7 * 24 * 3600
        self.data_filepath = data_filepath

        self.CAR_SERVICE_TIME_MEAN = 12.0 
        self.CAR_SERVICE_TIME_STD = 8.0

        self.TRUCK_SERVICE_TIME_MEAN = 30.0
        self.TRUCK_SERVICE_TIME_STD = 11.0

        self.TOLL_PER_CAR = 100
        self.TOLL_PER_TRUCK = 210

        self.LANE_TYPES = {'general': 0, 'car_only': 1, 'truck_only': 2}
        self.VEHICLE_TYPES = {'car': 0, 'truck': 1}

        num_lane_types = len(self.LANE_TYPES)
        self.action_space = spaces.Discrete(num_lane_types**self.num_lanes)
        self.observation_space = spaces.Box(low=0, high=200, 
                                            shape=(self.num_lanes, 2), dtype=np.int32)
    def _map_action_to_config(self, action):
        """Converts a single action integer (e.g., 13) into a lane configuration array (e.g., [1,1,0,1])."""
        config = []
        num_lane_types = len(self.LANE_TYPES)
        temp_action = action
        for _ in range(self.num_lanes):
            config.append(temp_action % num_lane_types)
            temp_action //= num_lane_types
        return np.array(config)

    def reset(self, seed=None, options=None):
        """Prepares the environment for a new episode (a new week of simulation)."""
        super().reset(seed=seed)
        self.queues = np.zeros((self.num_lanes, 2), dtype=np.int32)
        self.simulation_time = 0
        
        # Load the traffic data for the new simulation run
        self.arrival_events = traffic_generator.load_traffic_events_from_file(self.data_filepath)
        
        # Set simulation duration based on the loaded data
        if self.arrival_events:
            self.sim_duration_seconds = self.arrival_events[-1]['time']
        else:
            self.sim_duration_seconds = 0
            
        self.next_event_index = 0
        # Reset performance metrics
        self.total_revenue = 0.0
        self.hourly_throughput = {}
        self.total_wait_time_steps = 0
        self.total_vehicles_processed = 0
        
        return self.queues, {}

    def step(self, action):
        lane_config = self._map_action_to_config(action)

        # 1. Process Vehicle Arrivals
        start_time = self.simulation_time
        end_time = self.simulation_time + self.time_per_step
        while (self.next_event_index < len(self.arrival_events) and
               self.arrival_events[self.next_event_index]['time'] < end_time):
            event = self.arrival_events[self.next_event_index]
            v_type = self.VEHICLE_TYPES[event['type']]
            valid_lanes = [i for i, l_type in enumerate(lane_config) if l_type == 0 or l_type == v_type + 1]
            if valid_lanes:
                lane_queues = [np.sum(self.queues[i]) for i in valid_lanes]
                chosen_lane = valid_lanes[np.argmin(lane_queues)]
                self.queues[chosen_lane, v_type] += 1
            self.next_event_index += 1

        # 2. Process Vehicles from Queues
        cars_processed_total = 0
        trucks_processed_total = 0
        
        for i in range(self.num_lanes):
            lane_type = lane_config[i]
            time_budget = self.time_per_step
            while time_budget > 0:
                processed_vehicle = False
                # Process trucks
                if (lane_type == self.LANE_TYPES['general'] or lane_type == self.LANE_TYPES['truck_only']) and self.queues[i, 1] > 0:
                    # CHANGED: Generate service time from a Normal distribution
                    service_time = max(3, np.random.normal(self.TRUCK_SERVICE_TIME_MEAN, self.TRUCK_SERVICE_TIME_STD))
                    if time_budget >= service_time:
                        self.queues[i, 1] -= 1
                        trucks_processed_total += 1
                        time_budget -= service_time
                        processed_vehicle = True
                # Process cars
                elif (lane_type == self.LANE_TYPES['general'] or lane_type == self.LANE_TYPES['car_only']) and self.queues[i, 0] > 0:
                    # CHANGED: Generate service time from a Normal distribution
                    service_time = max(2, np.random.normal(self.CAR_SERVICE_TIME_MEAN, self.CAR_SERVICE_TIME_STD))
                    if time_budget >= service_time:
                        self.queues[i, 0] -= 1
                        cars_processed_total += 1
                        time_budget -= service_time
                        processed_vehicle = True
                
                if not processed_vehicle:
                    break 

        # 3. Update Performance Metrics
        self.total_revenue += (cars_processed_total * self.TOLL_PER_CAR) + (trucks_processed_total * self.TOLL_PER_TRUCK)
        current_hour = self.simulation_time // 3600
        self.hourly_throughput[current_hour] = self.hourly_throughput.get(current_hour, 0) + cars_processed_total + trucks_processed_total
        
        self.total_vehicles_processed += cars_processed_total + trucks_processed_total
        vehicles_still_waiting = np.sum(self.queues)
        self.total_wait_time_steps += vehicles_still_waiting

        # 4. Calculate Reward for the Agent
        reward_throughput = (cars_processed_total * 1.0) + (trucks_processed_total * 2.5)
        # Using the direct count of waiting vehicles in the penalty
        penalty_waiting = (np.sum(self.queues[:, 0]) * 0.1) + (np.sum(self.queues[:, 1]) * 0.3)
        reward = reward_throughput - penalty_waiting

        # 5. Conclude the Step
        self.simulation_time = end_time
        terminated = self.simulation_time >= self.sim_duration_seconds
        truncated = False
        
        # UPDATED: Add the new average wait time metric to the info dictionary
        avg_wait_time = (self.total_wait_time_steps * self.time_per_step) / self.total_vehicles_processed if self.total_vehicles_processed > 0 else 0
        info = {
            'revenue': self.total_revenue, 
            'hourly_throughput': self.hourly_throughput,
            'avg_wait_time_seconds': avg_wait_time
        }
        
        return self.queues, reward, terminated, truncated, info