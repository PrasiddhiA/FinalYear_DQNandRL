# traffic_generator.py

import os
import csv
import random
import pandas as pd
from datetime import datetime, timedelta

# --- Configuration ---
OUTPUT_CSV_FILE = "weekly_traffic_data.csv"
START_DATE = "2025-10-06" # A Monday

# --- Helper Functions for Realistic Generation ---
def _get_vehicle_mix(hour):
    """
    Returns the probability of a vehicle being a car based on the hour.
    CHANGED: The daytime mix is now based on data from the provided research paper.
    """
    if 7 <= hour < 20:  # Daytime (7 AM to 8 PM)
        return 0.65  # 65% chance of being a car, based on Table 4 data.
    else:  # Nighttime (assumption: more trucks)
        return 0.40  # 40% chance of being a car

# --- Main Generator and Loader Functions ---
def generate_vehicle_traffic_data(start_date, filename):
    """
    Generates high-volume, realistic traffic data for one week and saves it to a CSV.
    """
    print(f"Generating one week of HIGH-VOLUME traffic data from {start_date}...")
    
    VEHICLE_TYPES = {"car": 1.0, "truck": 2.5}
    days = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
    hours = list(range(24))

    # CHANGED: Hourly counts scaled up ~10x to match real-world Mumbai traffic volume (~70,000 vehicles/day).
    base_hourly_counts = [
        600, 500, 400, 400, 600, 1000, 3500, 5500, 4500, 3000, 2500, 2200,
        2300, 2500, 2600, 3000, 5800, 6800, 5500, 3500, 2800, 2200, 1800, 1200
    ]

    # These multipliers adjust traffic volume per day
    weekly_multipliers = {
        "MON": 1.05, "TUE": 1.0, "WED": 1.0, "THU": 1.0, 
        "FRI": 1.2, "SAT": 1.0, "SUN": 0.95
    }

    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    records = []

    for day_index, day in enumerate(days):
        for hour in hours:
            avg_vehicles = int(base_hourly_counts[hour] * weekly_multipliers[day])
            car_probability = _get_vehicle_mix(hour)

            for _ in range(avg_vehicles):
                random_second = random.randint(0, 3599)
                timestamp = start_datetime + timedelta(days=day_index, hours=hour, seconds=random_second)

                if random.random() < car_probability:
                    vehicle_type = "car"
                else:
                    vehicle_type = "truck"
                
                records.append({
                    "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "VehicleType": vehicle_type,
                })

    df = pd.DataFrame(records)
    df.sort_values(by="Timestamp", inplace=True)
    df.to_csv(filename, index=False, columns=["Timestamp", "VehicleType"]) # Save only needed columns
    print(f"Data generated and saved to '{filename}'")
    return df

def load_traffic_events_from_file(filepath):
    """
    Loads the generated CSV and converts it into the event list format
    that the TollPlazaEnv can understand.
    """
    print(f"Loading traffic data from {filepath}...")
    events = []
    start_time = None
    time_format = "%Y-%m-%d %H:%M:%S"

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            timestamp = datetime.strptime(row['Timestamp'], time_format)
            if i == 0:
                start_time = timestamp
            relative_time = (timestamp - start_time).total_seconds()
            events.append({'time': int(relative_time), 'type': row['VehicleType']})

    print(f"Loaded {len(events)} vehicle events for the simulation.")
    return events

# --- Main Execution Block ---
if __name__ == "__main__":
    if os.path.exists(OUTPUT_CSV_FILE):
        print(f"Removing old data file: {OUTPUT_CSV_FILE}")
        os.remove(OUTPUT_CSV_FILE)
        
    generate_vehicle_traffic_data(start_date=START_DATE, filename=OUTPUT_CSV_FILE)