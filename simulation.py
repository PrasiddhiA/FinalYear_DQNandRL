import os
import pygame
import numpy as np
from stable_baselines3 import DQN
from toll_plaza_env import TollPlazaEnv

# --- Config ---
models_dir = "models"
model_path = os.path.join(models_dir, "dqn_toll_plaza_final_model.zip")

# Load environment and trained model
env = TollPlazaEnv(num_lanes=4, data_filepath="weekly_traffic_data.csv")
model = DQN.load(model_path, env=env)

# --- Pygame Setup ---
pygame.init()
WIDTH, HEIGHT = 1000, 600
LANE_HEIGHT = 100
CAR_WIDTH, CAR_HEIGHT = 40, 25
BOOTH_WIDTH, BOOTH_HEIGHT = 20, 60
SPACING = 15  # space between queued cars
BOOTH_X = WIDTH - 100  # position of toll booths

win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Toll Plaza Queue Simulation 🚗")

# Colors
BLACK = (25, 25, 25)
WHITE = (255, 255, 255)
GRAY = (120, 120, 120)
RED = (200, 60, 60)
GREEN = (60, 200, 60)
YELLOW = (250, 200, 50)
BLUE = (80, 150, 255)

font = pygame.font.SysFont("Arial", 20)

# --- Car Object ---
class Car:
    def __init__(self, lane, vehicle_type):
        self.lane = lane
        self.vehicle_type = vehicle_type
        self.color = GREEN if vehicle_type == "car" else RED
        self.payment_timer = np.random.randint(30, 70)  # frames before passing booth
        self.x = 0

# --- Initialize Simulation ---
obs, info = env.reset()
clock = pygame.time.Clock()
running = True
done = False
step_count = 0

lanes = [[] for _ in range(env.num_lanes)]  # each lane is a list of Car objects

# --- Simulation Loop ---
while running and not done:
    clock.tick(30)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # RL action step
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    step_count += 1

    # Occasionally add new car into a random lane
    if np.random.rand() < 0.3:
        lane_idx = np.random.randint(0, env.num_lanes)
        vehicle_type = np.random.choice(["car", "truck"])
        lanes[lane_idx].append(Car(lane_idx, vehicle_type))

    # --- Update Queues ---
    for i, queue in enumerate(lanes):
        if not queue:
            continue

        for idx, car in enumerate(queue):
            target_x = BOOTH_X - (idx + 1) * (CAR_WIDTH + SPACING)

            # move car toward its target position smoothly
            if car.x < target_x:
                car.x += 3  # car speed
                if car.x > target_x:
                    car.x = target_x

        # Handle booth processing
        first_car = queue[0]
        if first_car.x >= BOOTH_X - CAR_WIDTH - 5:
            first_car.payment_timer -= 1
            if first_car.payment_timer <= 0:
                queue.pop(0)  # car leaves booth

    # --- Draw Scene ---
    win.fill(BLACK)

    # Draw road lanes
    for i in range(env.num_lanes):
        y = 100 + i * LANE_HEIGHT
        pygame.draw.line(win, GRAY, (0, y + LANE_HEIGHT // 2), (WIDTH, y + LANE_HEIGHT // 2), 2)

    # Draw toll booths
    for i in range(env.num_lanes):
        y = 100 + i * LANE_HEIGHT + LANE_HEIGHT // 4
        pygame.draw.rect(win, BLUE, (BOOTH_X, y, BOOTH_WIDTH, BOOTH_HEIGHT))

    # Draw cars in queue
    for i, queue in enumerate(lanes):
        y = 100 + i * LANE_HEIGHT + LANE_HEIGHT // 2 - CAR_HEIGHT // 2
        for car in queue:
            pygame.draw.rect(win, car.color, (car.x, y, CAR_WIDTH, CAR_HEIGHT))
            label = font.render(car.vehicle_type[0].upper(), True, BLACK)
            win.blit(label, (car.x + 10, y + 3))

    # --- Display Stats ---
    rev_text = font.render(f"Total Revenue: ₹{info.get('total_revenue', 0):.2f}", True, WHITE)
    step_text = font.render(f"Step: {step_count}", True, WHITE)
    reward_text = font.render(f"Reward: {reward:.2f}", True, YELLOW)
    win.blit(rev_text, (50, 30))
    win.blit(step_text, (400, 30))
    win.blit(reward_text, (700, 30))

    pygame.display.update()

pygame.quit()
print("\nSimulation complete.")
print(f"Total Revenue: ₹{info.get('total_revenue', 0):.2f}")
print(f"Total Steps: {step_count}")
