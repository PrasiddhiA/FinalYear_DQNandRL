# traffic_generator.py

import os
import csv
import random
import pandas as pd
from datetime import datetime, timedelta

# --- Configuration ---
OUTPUT_CSV_FILE = "weekly_traffic_data.csv"
START_DATE = "2025-10-06"  # A Monday

# --- Helper Functions ---
def _get_vehicle_mix(hour, day_index):
    """
    Returns probability of a vehicle being a car, 
    based on hour and weekday/weekend trends.
    """
    weekend = day_index >= 5  # Saturday, Sunday
    if 7 <= hour < 20:
        # Daytime: more cars than trucks (higher on weekends)
        return 0.7 if not weekend else 0.78
    else:
        # Nighttime: more trucks than cars (slightly fewer on weekends)
        return 0.4 if not weekend else 0.45


# --- Main Generator ---
def generate_vehicle_traffic_data(start_date, filename):
    """
    Generates realistic second-by-second toll traffic data for one week.
    """
    print(f"Generating one week of realistic traffic data from {start_date}...")

    VEHICLE_TYPES = {"car": 1.0, "truck": 2.5}
    days = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
    hours = list(range(24))

    # Approximate real-world hourly vehicle counts (~70k/day)
    base_hourly_counts = [
        600, 500, 400, 400, 600, 1000, 3500, 5500, 4500, 3000, 2500, 2200,
        2300, 2500, 2600, 3000, 5800, 6800, 5500, 3500, 2800, 2200, 1800, 1200
    ]

    # Adjust weekday/weekend traffic multipliers
    weekly_multipliers = {
        "MON": 1.05, "TUE": 1.0, "WED": 1.0, "THU": 1.0,
        "FRI": 1.2, "SAT": 1.1, "SUN": 0.9
    }

    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    records = []

    for day_index, day in enumerate(days):
        for hour in hours:
            # Apply weekday multiplier and small ±5% stochastic variation
            avg_vehicles = int(base_hourly_counts[hour] * weekly_multipliers[day] * random.uniform(0.95, 1.05))
            car_probability = _get_vehicle_mix(hour, day_index)

            for _ in range(avg_vehicles):
                random_second = random.randint(0, 3599)
                timestamp = start_datetime + timedelta(days=day_index, hours=hour, seconds=random_second)
                vehicle_type = "car" if random.random() < car_probability else "truck"

                records.append({
                    "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "VehicleType": vehicle_type,
                })

    
    df = pd.DataFrame(records)
    df.sort_values(by="Timestamp", inplace=True)
    df.to_csv(filename, index=False, columns=["Timestamp", "VehicleType"])
    print(f"✅ Data generated and saved to '{filename}' ({len(df)} records).")
    return df


# --- Loader for Simulation ---
def load_traffic_events_from_file(filepath):
    """
    Loads CSV and converts into event list for toll plaza simulation.
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
            events.append({'time': int(relative_time), 'VehicleType': row['VehicleType']})

    print(f"Loaded {len(events)} vehicle events for the simulation.")
    return events


# --- Main Execution ---
if __name__ == "__main__":
    if os.path.exists(OUTPUT_CSV_FILE):
        print(f"Removing old data file: {OUTPUT_CSV_FILE}")
        os.remove(OUTPUT_CSV_FILE)

    generate_vehicle_traffic_data(start_date=START_DATE, filename=OUTPUT_CSV_FILE)
