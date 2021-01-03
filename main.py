import environment
import numpy as np
import random

# Create world
world = environment.world(num_sound_waves=1, obstacle_locations=np.array([[20, -40], [-10, -15], [-35, -45], [-40, 5]]), obstacle_diameters=np.array([20, 30, 16, 24]))
#world = environment.world(num_sound_waves=10)

# Set starting position
world.startposition()

# Run one trial
while (not world.timeup() and not world.atgoal()):

    # Select random actions
    A = np.random.randint(0, 8)

    # Move agent
    world.move(A)

    # Show waves
    world.generate_waves(length=60)

# Plot path
world.plotpath()