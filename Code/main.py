import environment
import numpy as np
import random
import place_cell
import time
import matplotlib.pyplot as plt
import random
import math
import actor_critic

plt.rcParams["figure.figsize"] = (7,7)


def gen_ran_coords(radius, verbose=True):
    """
    Function for generating random coordinates in world.
    -- radius: world radius.
    """

    a = 2 * random.random() * math.pi
    r = radius * math.sqrt(random.random())

    x = r * math.cos(a)
    y = r * math.sin(a)

    if (verbose == True):
        print("Platform location: {}, {}".format(x, y))

    return np.array([x, y])


def gen_obs_coords(radius, pf_coords, pf_radius, obs_diams=np.array([]), verbose=False):
    """
    Function for generating obstacle locations randomly.
    -- radius: world radius
    -- pf_coords: platform coordinates
    -- pf_radius: platform radius
    -- obs_diams: obstacle diamaters 
    -- verbose: print statements or not
    """

    if (len(obs_diams) == 0):
        obstacle_locations = np.array([])

        return obstacle_locations

    else:
        obstacle_locations = np.full((len(obs_diams), 2), fill_value=[-math.inf, -math.inf])

        for k in range(len(obs_diams)):
            check = False

            while (check == False):

                check = True

                diam = obs_diams[k]
                x, y = gen_ran_coords(radius - 3/2 * diam, verbose=False)

                if ((pf_coords[0] - pf_radius - diam <= x <= pf_coords[0] + pf_radius) and
                    (pf_coords[1] - pf_radius - diam <= y <= pf_coords[1] + pf_radius)):

                    check = False
                    continue

                if (k == 0):
                    continue

                for l in range(k):
                    x_temp, y_temp = obstacle_locations[l]
                    diam_temp = obs_diams[l]

                    if ((x + diam < x_temp or x > x_temp + diam_temp) and 
                        (y + diam < y_temp or y > y_temp + diam_temp)):

                        continue
                    
                    else:
                        check = False

            obstacle_locations[k] = [x, y]

        return obstacle_locations


# Initialize everything
world_radius = 60
lr           = 0.5
gamma        = 0.99
num_trial    = 100
trial_len    = 100
num_days     = 36
num_cells    = 493
cell_std     = 3
pf_radius    = 10
path_len     = 100
num_dirs     = 8

obstacle_diameters=np.array([23, 15, 10, 8])

avg_path_len = np.zeros(num_days)
avg_path_std = np.zeros(num_days)

pf_change   = False
change_days = 4

# Algorithm options:
# -- td    : regular td-learning
# -- coord : td-learning with coordinate system
# -- both  : run both algorithms to compare performance
algorithm   = "ahahhaha"

# Set starting position randomly
pf_good = False
pf_coords = gen_ran_coords(world_radius - pf_radius)

obstacle_locations = gen_obs_coords(world_radius, pf_coords, pf_radius, obstacle_diameters)

pc_arr = place_cell.place_cells(num_cells, cell_std, world_radius, verbose=False)

# Start trials
if (algorithm == "td"):
    actor  = actor_critic.Actor(lr, num_dirs, num_cells)
    critic = actor_critic.Critic(lr, gamma, num_cells)

    # Set start position
    x_coord, y_coord = gen_ran_coords(world_radius, verbose=True)

    for day in range(num_days):
        path_len = []

        # Re-set start position
        if (pf_change == True and day % change_days == 0):
            x_coord, y_coord = gen_ran_coords(world_radius, verbose=True)

        for trial in num_trial:

            # Create world and its start position
            world  = environment.world(
                           T=path_len,
                           world_radius=world_radius, 
                           num_sound_waves=0, 
                           platform_location=pf_coords,
                           platform_radius=pf_radius,
                           obstacle_locations=obstacle_locations, 
                           obstacle_diameters=obstacle_diameters)
            
            world.startposition()

            # Run trial
            while (not world.timeup() and not world.atgoal()):
                
                # Determine new position from 
                position  = np.array(world.position[:, world.t])
                probs     = actor.probs(pc_arr)
                possibs   = np.linspace(0, num_dirs - 1, num=num_dirs, dtype=int)
                direction = np.random.choice(possibs, p=probs)
                world.move(direction)
                new_pos   = np.array(world.position[:, world.t])

                # Activate place cells
                for x in range(len(pc_arr)):
                    pc_arr[x].activate(new_pos)

                # Determine reward
                if (world.atgoal()):
                    rt = 1
                
                else:
                    rt = 0

                # Update weights
                error = critic.weight_update(rt, pc_arr)
                actor.weight_update(direction, error, pc_arr)
            
            path_len.append(world.t)
        
        arr_path_len  = np.asarray(path_len)
        mean_path_len = np.mean(arr_path_len)
        std_path_len  = np.std(arr_path_len)

        avg_path_len[day] = mean_path_len
        avg_path_std[day] = std_path_len

        print("Day {} -- Mean Escape Latency: {} +/- {}".format(day, mean_path_len, std_path_len))

    plt.errorbar(np.linspace(1, num_days, num=num_days), avg_path_len, avg_path_std, linestyle='dotted', marker='o', capsize=5)
    plt.xlabel("Day")
    plt.ylabel("Escape Latency (s)")
    plt.title("Pure TD Learning Escape Latency")
    plt.show()

else:

    # Create world and its start position
    world  = environment.world(
                T=path_len,
                world_radius=world_radius, 
                num_sound_waves=0, 
                platform_location=pf_coords,
                platform_radius=pf_radius,
                obstacle_locations=obstacle_locations, 
                obstacle_diameters=obstacle_diameters)

    world.startposition()

    # Run one trial
    while (not world.timeup() and not world.atgoal()):

        # Select random actions
        A = np.random.randint(0, num_dirs)

        # Move agent
        world.move(A)

        # Show waves
        world.generate_waves(length=60, verbose=False)

    # Plot path
    world.plotpath()