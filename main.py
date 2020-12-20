import environment
import numpy as np
import random

# Create world
world = environment.world()

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