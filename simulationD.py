import pygame
import random
import time
import numpy as np

# --- Pygame Setup ---
pygame.init()
WIDTH, HEIGHT = 1200, 600
LANE_COUNT = 4
BOOTH_COUNT = LANE_COUNT
BOOTH_X = WIDTH - 150
CAR_COLOR = (0, 200, 0)
TRUCK_COLOR = (200, 50, 50)
ROAD_COLOR = (40, 40, 40)
LINE_COLOR = (200, 200, 200)
FONT = pygame.font.SysFont("Arial", 20)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Toll Plaza Simulation 🚗🚛")

# --- Service Times ---
CAR_SERVICE_TIME_MEAN = 12.0
CAR_SERVICE_TIME_STD = 8.0
TRUCK_SERVICE_TIME_MEAN = 30.0
TRUCK_SERVICE_TIME_STD = 11.0

# --- Vehicle Class ---
class Vehicle:
    def __init__(self, vtype, lane, y):
        self.type = vtype
        self.lane = lane
        self.width = 40 if vtype == "car" else 70
        self.height = 30
        self.x = random.randint(0, WIDTH // 4)
        self.y = y
        self.color = CAR_COLOR if vtype == "car" else TRUCK_COLOR
        self.speed = random.uniform(1.5, 2.5) if vtype == "car" else random.uniform(1.0, 1.8)
        self.waiting = False
        self.service_time = None
        self.time_remaining = None

    def move(self):
        if not self.waiting:
            self.x += self.speed

# --- Create Vehicles ---
def spawn_vehicles_per_minute():
    """Spawns 3 cars + 2 trucks per minute distributed across lanes."""
    vehicles = []
    for _ in range(3):
        lane = random.randint(0, LANE_COUNT - 1)
        y = lane * (HEIGHT // LANE_COUNT) + 50
        vehicles.append(Vehicle("car", lane, y))
    for _ in range(2):
        lane = random.randint(0, LANE_COUNT - 1)
        y = lane * (HEIGHT // LANE_COUNT) + 50
        vehicles.append(Vehicle("truck", lane, y))
    return vehicles

# --- Simulation State ---
vehicles = []
spawn_timer = 0
running = True
clock = pygame.time.Clock()
fps = 30
total_revenue = 0
step = 0

# --- Main Loop ---
while running:
    clock.tick(fps)
    screen.fill((20, 20, 20))

    # Draw road lanes
    for i in range(LANE_COUNT):
        y = i * (HEIGHT // LANE_COUNT)
        pygame.draw.rect(screen, ROAD_COLOR, (0, y, WIDTH, HEIGHT // LANE_COUNT - 5))
        pygame.draw.line(screen, LINE_COLOR, (0, y), (WIDTH, y), 2)
    pygame.draw.line(screen, LINE_COLOR, (0, HEIGHT - 5), (WIDTH, HEIGHT - 5), 2)

    # Draw toll booths
    for i in range(BOOTH_COUNT):
        booth_y = i * (HEIGHT // LANE_COUNT) + (HEIGHT // LANE_COUNT) // 2 - 25
        pygame.draw.rect(screen, (80, 80, 150), (BOOTH_X, booth_y, 20, 50))

    # Spawn new vehicles roughly once per simulated minute
    spawn_timer += 1
    if spawn_timer > fps * 3:  # ~3 seconds real-time = 1 minute simulated
        vehicles.extend(spawn_vehicles_per_minute())
        spawn_timer = 0

    # Move vehicles
    for vehicle in vehicles:
        if vehicle.x + vehicle.width >= BOOTH_X - 10:
            if not vehicle.waiting:
                vehicle.waiting = True
                if vehicle.type == "car":
                    service_time = max(1, np.random.normal(CAR_SERVICE_TIME_MEAN, CAR_SERVICE_TIME_STD))
                else:
                    service_time = max(1, np.random.normal(TRUCK_SERVICE_TIME_MEAN, TRUCK_SERVICE_TIME_STD))
                vehicle.time_remaining = service_time
                vehicle.service_time = time.time()
            else:
                elapsed = time.time() - vehicle.service_time
                if elapsed >= vehicle.time_remaining:
                    total_revenue += 100 if vehicle.type == "car" else 250
                    vehicles.remove(vehicle)
                    continue
        else:
            # Prevent overlap (queue effect)
            same_lane = [v for v in vehicles if v.lane == vehicle.lane and v != vehicle]
            ahead = [v for v in same_lane if v.x > vehicle.x]
            if ahead:
                nearest = min(ahead, key=lambda v: v.x)
                if nearest.x - vehicle.x < vehicle.width + 10:
                    continue  # stop moving (keep distance)
            vehicle.move()

        # Draw vehicle
        pygame.draw.rect(screen, vehicle.color, (vehicle.x, vehicle.y, vehicle.width, vehicle.height))
        vtext = FONT.render("C" if vehicle.type == "car" else "T", True, (0, 0, 0))
        screen.blit(vtext, (vehicle.x + 10, vehicle.y + 5))

    # Display Stats
    step += 1
    text_rev = FONT.render(f"Total Revenue: ₹{total_revenue:.2f}", True, (255, 255, 255))
    text_step = FONT.render(f"Step: {step}", True, (255, 255, 255))
    screen.blit(text_rev, (50, 20))
    screen.blit(text_step, (WIDTH - 250, 20))

    # Quit Event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()

pygame.quit()
