import environment
import numpy as np
import random

# Create world
world = environment.world(obstacle_locations=np.array([[20, -40], [-10, -15], [-35, -45], [-40, 5]]), obstacle_diameters=np.array([20, 30, 16, 24]))

# Set starting position
world.startposition()

# Run one trial
while (not world.timeup() and not world.atgoal()):

    # Select random actions
    A = np.random.randint(0, 8)

    # Move agent
    world.move(A)

# Plot path
world.plotpath()