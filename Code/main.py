import environment
import numpy as np
import random
import place_cell
import time
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (7,7)

# Create world
world_radius = 60

world = environment.world(world_radius=world_radius, num_sound_waves=1, obstacle_locations=np.array([[20, -40], [-10, -15], [-35, -45], [-40, 5]]), obstacle_diameters=np.array([20, 30, 16, 24]))
pc_arr = place_cell.place_cells(493, 3, world_radius, verbose=True)

# Set starting position
world.startposition()

# Run one trial
while (not world.timeup() and not world.atgoal()):

    # Select random actions
    A = np.random.randint(0, 8)

    # Move agent
    world.move(A)

    # Show waves
    world.generate_waves(length=60, verbose=False)

# Plot path
world.plotpath()