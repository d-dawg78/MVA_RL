import environment
import numpy as np
import random
import place_cell
import time
import matplotlib.pyplot as plt
import random
import math
import actor_critic
import coordinate_system

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


def process_waves(angles, directions, probs):
    """
    Function for rebalancing probabilities according to waves received.
    -- angles: dictionary of waves with status and hit direction
    -- directions: all angle directions
    """
    b = 2 * np.pi / len(directions)
    modified = False

    probas = probs.copy()

    for _ , (status, ang) in angles.items():

        # No information
        if (status == 0):
            continue

        # Hit wall or obstacle
        elif (status == 1):
            modified = True

            upper = ang + b
            down  = ang - b

            for move, direc in directions.items():
                if (upper <= direc <= down or down <= direc <= upper):
                    probas[move] = 0.
                else:
                    continue

        elif (status == 2):
            modified = True

            upper = ang + b
            down  = ang - b

            #print("Upper {} and downer {}".format(upper, down))

            for move, direc in directions.items():
                if (upper <= direc <= down or down <= direc <= upper):
                    # for regular td: probas[move] = 1.
                    probas[move] = probas[move]
                else:
                    probas[move] = 0.

            if (probas[0:len(directions)].sum(0) < 0.000000000000001):
                probas[0:len(directions)] = 0.
            else:
                probas[0:len(directions)] = (probas[0:len(directions)] / probas[0:len(directions)].sum(0)) * (1 - probas[len(directions)])

            break

        else:
            print("BUG: status is not 0, 1, or 2.")
    
    if (probas.sum(0) < 0.000000000000001):
        print("!!! PROBABILITIES SUM TO 0: {} !!!".format(probas))
        move = np.random.randint(0, num_dirs)
        probas[move] = 1.

    probas = probas / probas.sum(0)

    #if (modified == True):
    #    print("Modified array from {} to {}".format(probs, probas))

    return probas


# Initialize everything
world_radius = 60
lr           = 0.5
gamma        = 0.99
num_trial    = 100
num_days     = 32
num_cells    = 493
cell_std     = 3
pf_radius    = 10
path_len     = 60
num_dirs     = 8
lam          = 0.9
num_waves    = 10

#obstacle_diameters=np.array([23, 15, 10, 8])
obstacle_diameters=np.array([])

pf_change   = True
change_days = 8

# Algorithm options:
# -- td    : regular td-learning
# -- coord : td-learning with coordinate system
# -- both  : run both algorithms to compare performance
algorithm   = "coord"

# Set starting position randomly
pf_good = False

pc_arr1 = place_cell.place_cells(num_cells, cell_std, world_radius, verbose=False)
pc_arr2 = place_cell.place_cells(num_cells, cell_std, world_radius, verbose=False)


# Start trials
if (algorithm == "td"):
    actor_td  = actor_critic.Actor(lr, num_dirs, num_cells)
    critic_td = actor_critic.Critic(lr, gamma, num_cells)

    avg_path_len_td = np.zeros(num_days)
    avg_path_std_td = np.zeros(num_days)
    avg_coli_len_td = np.zeros(num_days)
    avg_coli_std_td = np.zeros(num_days)

    actor  = actor_critic.Actor(lr, num_dirs, num_cells)
    critic = actor_critic.Critic(lr, gamma, num_cells)

    avg_path_len = np.zeros(num_days)
    avg_path_std = np.zeros(num_days)
    avg_coli_len = np.zeros(num_days)
    avg_coli_std = np.zeros(num_days)

    # Set start position
    pf_coords = gen_ran_coords(world_radius - pf_radius)
    obstacle_locations = gen_obs_coords(world_radius, pf_coords, pf_radius, obstacle_diameters)
    x_coord, y_coord = pf_coords

    for day in range(num_days):
        arr_path_len_td = []
        num_colision_td = []

        arr_path_len = []
        num_colision = []

        # Re-set start position
        if (pf_change == True and day % change_days == 0 and day != 0):

            # Set start position
            pf_coords = gen_ran_coords(world_radius - pf_radius)
            obstacle_locations = gen_obs_coords(world_radius, pf_coords, pf_radius, obstacle_diameters)
            x_coord, y_coord = pf_coords

        for trial in range(num_trial):

            # Create world and its start position
            world_td  = environment.world(
                           T=path_len,
                           world_radius=world_radius, 
                           num_directions=num_dirs,
                           num_sound_waves=num_waves, 
                           platform_location=pf_coords,
                           platform_radius=pf_radius,
                           obstacle_locations=obstacle_locations, 
                           obstacle_diameters=obstacle_diameters)
            
            world_td.startposition()

            # Create world and its start position
            world  = environment.world(
                           T=path_len,
                           world_radius=world_radius, 
                           num_directions=num_dirs,
                           num_sound_waves=num_waves, 
                           platform_location=pf_coords,
                           platform_radius=pf_radius,
                           obstacle_locations=obstacle_locations, 
                           obstacle_diameters=obstacle_diameters)
            
            world.startposition()

            # Run trial
            while (not world_td.timeup() and not world_td.atgoal()):
                
                # Determine new position 
                position_td  = np.array(world_td.position[:, world_td.t])

                # Get soundwaves
                angles = world_td.generate_waves(length=60, verbose=False)

                # Get probabilities and re-balance
                probs_td     = actor_td.probs(pc_arr2)
                new_probs_td = process_waves(angles, world_td.direction, probs_td)

                possibs_td   = np.linspace(0, num_dirs - 1, num=num_dirs, dtype=int)
                direction_td = np.random.choice(possibs_td, p=new_probs_td)
                world_td.move(direction_td)
                new_pos_td   = np.array(world_td.position[:, world_td.t])

                # Activate place cells
                for x in range(len(pc_arr2)):
                    pc_arr2[x].activate(new_pos_td)

                # Determine reward
                if (world_td.atgoal()):
                    rt = 1
                
                else:
                    rt = 0

                # Update weights
                error_td = critic_td.weight_update(rt, pc_arr2)
                actor_td.weight_update(direction_td, error_td, pc_arr2)

            # Run trial
            while (not world.timeup() and not world.atgoal()):
                
                # Determine new position 
                position  = np.array(world.position[:, world.t])

                # Get probabilities and re-balance
                probs     = actor.probs(pc_arr1)
                possibs   = np.linspace(0, num_dirs - 1, num=num_dirs, dtype=int)
                direction = np.random.choice(possibs, p=probs)
                world.move(direction)
                new_pos   = np.array(world.position[:, world.t])

                # Activate place cells
                for x in range(len(pc_arr1)):
                    pc_arr1[x].activate(new_pos)

                # Determine reward
                if (world.atgoal()):
                    rt = 1
                
                else:
                    rt = 0

                # Update weights
                error = critic.weight_update(rt, pc_arr1)
                actor.weight_update(direction, error, pc_arr1)
            
            print("\nNormal Trial   : {}. Time: {}. Collisions: {}.".format(trial, world.t, world.colisions))
            print("Sounds Trial   : {}. Time: {}. Collisions: {}.".format(trial, world_td.t, world_td.colisions))
            #print("Colisions: {} / s".format(world.colisions))
            arr_path_len_td.append(world_td.t)
            num_colision_td.append(world_td.colisions)
            arr_path_len.append(world.t)
            num_colision.append(world.colisions)

            #world.plotpath()
        
        # Soundwave
        arr_path_len_td  = np.asarray(arr_path_len_td)
        mean_path_len_td = np.mean(arr_path_len_td)
        std_path_len_td  = np.std(arr_path_len_td)

        arr_coli_len_td  = np.asarray(num_colision_td)
        mean_coli_len_td = np.mean(arr_coli_len_td)
        std_coli_len_td  = np.std(arr_coli_len_td)

        avg_path_len_td[day] = mean_path_len_td
        avg_path_std_td[day] = std_path_len_td

        avg_coli_len_td[day] = mean_coli_len_td
        avg_coli_std_td[day] = std_coli_len_td

        # Normal
        arr_path_len  = np.asarray(arr_path_len)
        mean_path_len = np.mean(arr_path_len)
        std_path_len  = np.std(arr_path_len)

        arr_coli_len  = np.asarray(num_colision)
        mean_coli_len = np.mean(arr_coli_len)
        std_coli_len  = np.std(arr_coli_len)

        avg_path_len[day] = mean_path_len
        avg_path_std[day] = std_path_len

        avg_coli_len[day] = mean_coli_len
        avg_coli_std[day] = std_coli_len

        print("\nDay {} -- Normal Mean Escape Latency   : {} +/- {}".format(day, mean_path_len, std_path_len))
        print("Day {} -- Sounds Mean Escape Latency: {} +/- {}".format(day, mean_path_len_td, std_path_len_td))
        print("Day {} -- Normal Average colision count   : {} +/- {}".format(day, mean_coli_len, std_coli_len))
        print("Day {} -- Sounds Average colision count: {} +/- {}".format(day, mean_coli_len_td, std_coli_len_td))

    plt.errorbar(np.linspace(1, num_days, num=num_days), avg_path_len, avg_path_std, color='green', linestyle='dotted', marker='o', capsize=5)
    plt.errorbar(np.linspace(1, num_days, num=num_days), avg_path_len_td, avg_path_std_td, color='magenta', linestyle='dotted', marker='o', capsize=5)
    plt.legend(['No Soundwaves', 'Soundwaves'])
    plt.xlabel("Day")
    plt.ylabel("Escape Latency (s)")
    plt.title("Pure TD Learning Mean Escape Latency per Day (Regular Watermaze)")
    plt.show()

    plt.errorbar(np.linspace(1, num_days, num=num_days), avg_coli_len, avg_coli_std, color='green', linestyle='dotted', marker='o', capsize=5)
    plt.errorbar(np.linspace(1, num_days, num=num_days), avg_coli_len_td, avg_coli_std_td, color='magenta', linestyle='dotted', marker='o', capsize=5)
    plt.legend(['No Soundwaves', 'Soundwaves'])
    plt.xlabel("Day")
    plt.ylabel("Collision Count")
    plt.title("Pure TD Learning Mean Collision Count per Day (Regular Watermaze)")
    plt.show()


elif (algorithm == "coord"):
    actor  = coordinate_system.Actor(lr, num_dirs, num_cells)
    critic = coordinate_system.Critic(lr, gamma, num_cells)
    coords = coordinate_system.Coord_System(lr, lam, num_cells)

    avg_path_len = np.zeros(num_days)
    avg_path_std = np.zeros(num_days)
    avg_coli_len = np.zeros(num_days)
    avg_coli_std = np.zeros(num_days)

    # Set start position
    pf_coords = gen_ran_coords(world_radius - pf_radius)
    obstacle_locations = gen_obs_coords(world_radius, pf_coords, pf_radius, obstacle_diameters)
    x_coord, y_coord = pf_coords

    goal_memory  = np.array([0. ,0.])
    goal_learned = False
    goal_reached = 0

    temp_fr = np.zeros(num_cells)

    actor1  = coordinate_system.Actor(lr, num_dirs, num_cells)
    critic1 = coordinate_system.Critic(lr, gamma, num_cells)
    coords1 = coordinate_system.Coord_System(lr, lam, num_cells)

    avg_path_len1 = np.zeros(num_days)
    avg_path_std1 = np.zeros(num_days)
    avg_coli_len1 = np.zeros(num_days)
    avg_coli_std1 = np.zeros(num_days)

    goal_memory1  = np.array([0. ,0.])
    goal_learned1 = False
    goal_reached1 = 0

    temp_fr1 = np.zeros(num_cells)

    for day in range(num_days):
        arr_path_len = []
        num_colision = []

        arr_path_len1 = []
        num_colision1 = []

        # Re-set start position
        if (pf_change == True and day % change_days == 0 and day != 0):

            # Set start position
            pf_coords = gen_ran_coords(world_radius - pf_radius)
            obstacle_locations = gen_obs_coords(world_radius, pf_coords, pf_radius, obstacle_diameters)
            x_coord, y_coord = pf_coords

        for trial in range(num_trial):
            coords.fps = []
            coords1.fps = []

            # Create world and its start position
            world  = environment.world(
                           T=path_len,
                           world_radius=world_radius, 
                           num_directions=num_dirs,
                           num_sound_waves=0, 
                           platform_location=pf_coords,
                           platform_radius=pf_radius,
                           obstacle_locations=obstacle_locations, 
                           obstacle_diameters=obstacle_diameters)
            
            world.startposition()

            # Create world and its start position
            world1  = environment.world(
                           T=path_len,
                           world_radius=world_radius, 
                           num_directions=num_dirs,
                           num_sound_waves=num_waves, 
                           platform_location=pf_coords,
                           platform_radius=pf_radius,
                           obstacle_locations=obstacle_locations, 
                           obstacle_diameters=obstacle_diameters)
            
            world1.startposition()

            X_p = 0
            Y_p = 0

            X_p1 = 0
            Y_p1 = 0

            # Run trial
            while (not world1.timeup() and not world1.atgoal()):

                update_coord = False
                
                # Determine new position from 
                position  = np.array(world1.position[:, world1.t])

                # Get soundwaves
                angles = world1.generate_waves(length=60, verbose=False)

                # Get probabilities and re-balance
                probs     = actor1.probs(pc_arr1)
                new_probs = process_waves(angles, world1.direction, probs)

                possibs   = np.linspace(0, num_dirs, num=num_dirs + 1, dtype=int)
                direction = np.random.choice(possibs, p=new_probs)

                if (direction == num_dirs):

                    # Random move when platform location has not been learned
                    if (goal_learned1 == False):
                        direc = random.randint(0, num_dirs - 1)
                        update_coord = False

                    else:
                        c_probs    = np.asarray([X_p1, Y_p1])
                        est_dir    = goal_memory1 - c_probs
                        future_pos = np.transpose(np.asarray([world1.future_position(x) for x in range(num_dirs)]))
                        similarity = np.dot(est_dir[:], future_pos) / (np.linalg.norm(est_dir[:]) * np.linalg.norm(future_pos))
                        direc      = np.argmax(similarity)
                        update_coord = True

                        #print("!!! Chose coordinate system. Est_dir: {}. Chosen_dir: {} !!!".format(est_dir, direc))

                    world1.move(direc)

                else:
                    world1.move(direction)

                new_pos   = np.array(world1.position[:, world1.t])

                # Activate place cells
                for x in range(len(pc_arr1)):
                    pc_arr1[x].activate(new_pos)

                for x in range(len(temp_fr)):
                    temp_fr1[x] = pc_arr1[x].prev

                coords1.fps.append(temp_fr1)

                # Determine reward
                if (world1.atgoal()):
                    rt = 1

                    if (abs(new_pos[0] - goal_memory1[0]) < pf_radius * 2 and abs(new_pos[1] - goal_memory1[1]) < pf_radius * 2):
                        goal_memory1[:] = (goal_memory1[:] * goal_reached1 + new_pos[:]) / (goal_reached1 + 1)
                        goal_reached1  += 1
                        goal_learned1   = True
                        #print("--> {} : Weighted Goal Learned: {}, {}".format(trial, goal_memory[0], goal_memory[1]))

                    else:
                        goal_reached1 = 1
                        goal_memory1[:] = new_pos[:]
                        goal_learned1 = True
                        #print("--> {} : New Goal Learned: {}, {}".format(trial, goal_memory[0], goal_memory[1]))

                else:
                    rt = 0

                # Update weights
                error = critic1.weight_update(rt, pc_arr1)
                X_curr, X_prev, Y_curr, Y_prev = coords1.probs(pc_arr1)
                actor1.weight_update(direction, error, pc_arr1)

                delta_X = new_pos[0] - position[0]
                delta_Y = new_pos[1] - position[1]

                coords1.weight_update(pc_arr1, delta_X, delta_Y, X_curr, X_prev, Y_curr, Y_prev)

                X_p1 = X_curr
                Y_p1 = Y_curr

                if (update_coord == True):
                    actor1.coord_update(error)
            
            # Run trial
            while (not world.timeup() and not world.atgoal()):

                update_coord = False
                
                # Determine new position from 
                position  = np.array(world.position[:, world.t])
                probs     = actor.probs(pc_arr2)
                possibs   = np.linspace(0, num_dirs, num=num_dirs + 1, dtype=int)
                direction = np.random.choice(possibs, p=probs)

                if (direction == num_dirs):

                    # Random move when platform location has not been learned
                    if (goal_learned == False):
                        direc = random.randint(0, num_dirs - 1)
                        update_coord = False

                    else:
                        c_probs    = np.asarray([X_p, Y_p])
                        est_dir    = goal_memory - c_probs
                        future_pos = np.transpose(np.asarray([world.future_position(x) for x in range(num_dirs)]))
                        similarity = np.dot(est_dir[:], future_pos) / (np.linalg.norm(est_dir[:]) * np.linalg.norm(future_pos))
                        direc      = np.argmax(similarity)
                        update_coord = True

                        #print("!!! Chose coordinate system. Est_dir: {}. Chosen_dir: {} !!!".format(est_dir, direc))

                    world.move(direc)

                else:
                    world.move(direction)

                new_pos   = np.array(world.position[:, world.t])

                # Activate place cells
                for x in range(len(pc_arr2)):
                    pc_arr2[x].activate(new_pos)

                for x in range(len(temp_fr)):
                    temp_fr[x] = pc_arr2[x].prev

                coords.fps.append(temp_fr)

                # Determine reward
                if (world.atgoal()):
                    rt = 1

                    if (abs(new_pos[0] - goal_memory[0]) < pf_radius * 2 and abs(new_pos[1] - goal_memory[1]) < pf_radius * 2):
                        goal_memory[:] = (goal_memory[:] * goal_reached + new_pos[:]) / (goal_reached + 1)
                        goal_reached  += 1
                        goal_learned   = True
                        #print("--> {} : Weighted Goal Learned: {}, {}".format(trial, goal_memory[0], goal_memory[1]))

                    else:
                        goal_reached = 1
                        goal_memory[:] = new_pos[:]
                        goal_learned = True
                        #print("--> {} : New Goal Learned: {}, {}".format(trial, goal_memory[0], goal_memory[1]))

                else:
                    rt = 0

                # Update weights
                error = critic.weight_update(rt, pc_arr2)
                X_curr, X_prev, Y_curr, Y_prev = coords.probs(pc_arr2)
                actor.weight_update(direction, error, pc_arr2)

                delta_X = new_pos[0] - position[0]
                delta_Y = new_pos[1] - position[1]

                coords.weight_update(pc_arr2, delta_X, delta_Y, X_curr, X_prev, Y_curr, Y_prev)

                X_p = X_curr
                Y_p = Y_curr

                if (update_coord == True):
                    actor.coord_update(error)
            
            arr_path_len.append(world.t)
            arr_path_len1.append(world1.t)
            num_colision.append(world.colisions)
            num_colision1.append(world1.colisions)

            print("\nNormal Trial   : {}. Time: {}. Collisions: {}.".format(trial, world.t, world.colisions))
            print("Sounds Trial   : {}. Time: {}. Collisions: {}.".format(trial, world1.t, world1.colisions))
        
        # Normal
        arr_path_len  = np.asarray(arr_path_len)
        mean_path_len = np.mean(arr_path_len)
        std_path_len  = np.std(arr_path_len)

        avg_path_len[day] = mean_path_len
        avg_path_std[day] = std_path_len

        arr_coli_len  = np.asarray(num_colision)
        mean_coli_len = np.mean(arr_coli_len)
        std_coli_len  = np.std(arr_coli_len)

        avg_coli_len[day] = mean_coli_len
        avg_coli_std[day] = std_coli_len

        # Sounds
        arr_path_len1  = np.asarray(arr_path_len1)
        mean_path_len1 = np.mean(arr_path_len1)
        std_path_len1  = np.std(arr_path_len1)

        avg_path_len1[day] = mean_path_len1
        avg_path_std1[day] = std_path_len1

        arr_coli_len1  = np.asarray(num_colision1)
        mean_coli_len1 = np.mean(arr_coli_len1)
        std_coli_len1  = np.std(arr_coli_len1)

        avg_coli_len1[day] = mean_coli_len1
        avg_coli_std1[day] = std_coli_len1

        print("Day {} -- Coordinate Mean Escape Latency: {} +/- {}".format(day, mean_path_len, std_path_len))
        print("Day {} -- Soundwaves Mean Escape Latency: {} +/- {}".format(day, mean_path_len1, std_path_len1))
        print("Day {} -- Normal Average colision count : {} +/- {}".format(day, mean_coli_len, std_coli_len))
        print("Day {} -- Sounds Average colision count : {} +/- {}".format(day, mean_coli_len1, std_coli_len1))

    plt.errorbar(np.linspace(1, num_days, num=num_days), avg_path_len, avg_path_std, color='green', linestyle='dotted', marker='o', capsize=5)
    plt.errorbar(np.linspace(1, num_days, num=num_days), avg_path_len1, avg_path_std1, color='magenta', linestyle='dotted', marker='o', capsize=5)
    plt.legend(['No Soundwave', 'Soundwave'])
    plt.xlabel("Day")
    plt.ylabel("Escape Latency (s)")
    plt.title("Coordinate System Escape Latency")
    plt.show()

    plt.errorbar(np.linspace(1, num_days, num=num_days), avg_coli_len, avg_coli_std, color='green', linestyle='dotted', marker='o', capsize=5)
    plt.errorbar(np.linspace(1, num_days, num=num_days), avg_coli_len1, avg_coli_std1, color='magenta', linestyle='dotted', marker='o', capsize=5)
    plt.legend(['No Soundwaves', 'Soundwaves'])
    plt.xlabel("Day")
    plt.ylabel("Collision Count")
    plt.title("Pure TD Learning Mean Collision Count per Day (Regular Watermaze)")
    plt.show()


elif (algorithm == "both"):
    actor_coord  = coordinate_system.Actor(lr, num_dirs, num_cells)
    critic_coord = coordinate_system.Critic(lr, gamma, num_cells)
    coords = coordinate_system.Coord_System(lr, lam, num_cells)

    actor_td  = actor_critic.Actor(lr, num_dirs, num_cells)
    critic_td = actor_critic.Critic(lr, gamma, num_cells)

    avg_path_len_coords = np.zeros(num_days)
    avg_path_std_coords = np.zeros(num_days)

    avg_path_len_td = np.zeros(num_days)
    avg_path_std_td = np.zeros(num_days)

    # Set start position
    pf_coords = gen_ran_coords(world_radius - pf_radius)
    obstacle_locations = gen_obs_coords(world_radius, pf_coords, pf_radius, obstacle_diameters)
    x_coord, y_coord = pf_coords

    goal_memory  = np.array([0. ,0.])
    goal_learned = False
    goal_reached = 0

    temp_fr = np.zeros(num_cells)

    for day in range(num_days):
        arr_path_len_coords = []
        arr_path_len_td     = []

        # Re-set start position
        if (pf_change == True and day % change_days == 0 and day != 0):

            # Set start position
            pf_coords = gen_ran_coords(world_radius - pf_radius)
            obstacle_locations = gen_obs_coords(world_radius, pf_coords, pf_radius, obstacle_diameters)
            x_coord, y_coord = pf_coords

        for trial in range(num_trial):
            coords.fps = []

            # Create world and its start position
            world_coords  = environment.world(
                           T=path_len,
                           world_radius=world_radius, 
                           num_directions=num_dirs,
                           num_sound_waves=num_waves, 
                           platform_location=pf_coords,
                           platform_radius=pf_radius,
                           obstacle_locations=obstacle_locations, 
                           obstacle_diameters=obstacle_diameters)

            world_td  = environment.world(
                           T=path_len,
                           world_radius=world_radius, 
                           num_directions=num_dirs,
                           num_sound_waves=num_waves, 
                           platform_location=pf_coords,
                           platform_radius=pf_radius,
                           obstacle_locations=obstacle_locations, 
                           obstacle_diameters=obstacle_diameters)
            
            world_coords.startposition()
            world_td.startposition()

            X_p = 0
            Y_p = 0

            # Run coordinate trial
            while (not world_coords.timeup() and not world_coords.atgoal()):

                update_coord = False
                
                # Determine new position from 
                position  = np.array(world_coords.position[:, world_coords.t])
                probs     = actor_coord.probs(pc_arr2)
                possibs   = np.linspace(0, num_dirs, num=num_dirs + 1, dtype=int)
                direction = np.random.choice(possibs, p=probs)

                if (direction == num_dirs):

                    # Random move when platform location has not been learned
                    if (goal_learned == False):
                        direc = random.randint(0, num_dirs - 1)
                        update_coord = False

                    else:
                        c_probs    = np.asarray([X_p, Y_p])
                        est_dir    = goal_memory - c_probs
                        future_pos = np.transpose(np.asarray([world_coords.future_position(x) for x in range(num_dirs)]))
                        similarity = np.dot(est_dir[:], future_pos) / (np.linalg.norm(est_dir[:]) * np.linalg.norm(future_pos))
                        direc      = np.argmax(similarity)
                        update_coord = True

                        #print("!!! Chose coordinate system. Est_dir: {}. Chosen_dir: {} !!!".format(est_dir, direc))

                    world_coords.move(direc)

                else:
                    world_coords.move(direction)

                new_pos   = np.array(world_coords.position[:, world_coords.t])

                # Activate place cells
                for x in range(len(pc_arr2)):
                    pc_arr2[x].activate(new_pos)

                for x in range(len(temp_fr)):
                    temp_fr[x] = pc_arr2[x].prev

                coords.fps.append(temp_fr)

                # Determine reward
                if (world_coords.atgoal()):
                    rt = 1

                    if (abs(new_pos[0] - goal_memory[0]) < pf_radius * 2 and abs(new_pos[1] - goal_memory[1]) < pf_radius * 2):
                        goal_memory[:] = (goal_memory[:] * goal_reached + new_pos[:]) / (goal_reached + 1)
                        goal_reached  += 1
                        goal_learned   = True
                        print("--> {} : Weighted Goal Learned: {}, {}".format(trial, goal_memory[0], goal_memory[1]))

                    else:
                        goal_reached = 1
                        goal_memory[:] = new_pos[:]
                        goal_learned = True
                        print("--> {} : New Goal Learned: {}, {}".format(trial, goal_memory[0], goal_memory[1]))

                else:
                    rt = 0

                # Update weights
                error = critic_coord.weight_update(rt, pc_arr2)
                X_curr, X_prev, Y_curr, Y_prev = coords.probs(pc_arr2)
                actor_coord.weight_update(direction, error, pc_arr2)

                delta_X = new_pos[0] - position[0]
                delta_Y = new_pos[1] - position[1]

                coords.weight_update(pc_arr2, delta_X, delta_Y, X_curr, X_prev, Y_curr, Y_prev)

                X_p = X_curr
                Y_p = Y_curr

                if (update_coord == True):
                    actor_coord.coord_update(error)
            
            # Run td learning trial
            while (not world_td.timeup() and not world_td.atgoal()):
                
                # Determine new position from 
                position  = np.array(world_td.position[:, world_td.t])
                probs     = actor_td.probs(pc_arr1)
                possibs   = np.linspace(0, num_dirs - 1, num=num_dirs, dtype=int)
                direction = np.random.choice(possibs, p=probs)
                world_td.move(direction)
                new_pos   = np.array(world_td.position[:, world_td.t])

                # Activate place cells
                for x in range(len(pc_arr1)):
                    pc_arr1[x].activate(new_pos)

                # Determine reward
                if (world_td.atgoal()):
                    rt = 1
                
                else:
                    rt = 0

                # Update weights
                error = critic_td.weight_update(rt, pc_arr1)
                actor_td.weight_update(direction, error, pc_arr1)

            arr_path_len_coords.append(world_coords.t)
            arr_path_len_td.append(world_td.t)
        
        arr_path_len_coords  = np.asarray(arr_path_len_coords)
        mean_path_len_coords = np.mean(arr_path_len_coords)
        std_path_len_coords  = np.std(arr_path_len_coords)

        arr_path_len_td  = np.asarray(arr_path_len_td)
        mean_path_len_td = np.mean(arr_path_len_td)
        std_path_len_td  = np.std(arr_path_len_td)

        avg_path_len_coords[day] = mean_path_len_coords
        avg_path_std_coords[day] = std_path_len_coords

        avg_path_len_td[day] = mean_path_len_td
        avg_path_std_td[day] = std_path_len_td

        print("Day {} -- Pure TD-learning Mean Escape Latency : {} +/- {}".format(day, mean_path_len_td, std_path_len_td))
        print("Day {} -- Coordinate System Mean Escape Latency: {} +/- {}".format(day, mean_path_len_coords, std_path_len_coords))

    plt.errorbar(np.linspace(1, num_days, num=num_days), avg_path_len_td, avg_path_std_td, color='green', linestyle='dotted', marker='o', capsize=5)
    plt.errorbar(np.linspace(1, num_days, num=num_days), avg_path_len_coords, avg_path_std_coords, color='magenta', linestyle='dotted', marker='o', capsize=5)
    plt.legend(['Pure TD-learning', 'Coordinate System'])
    plt.xlabel("Day")
    plt.ylabel("Escape Latency (s)")
    plt.title("Algorithm Mean Escape Latency per Day")
    plt.show()


else:

    # Set start position
    pf_coords = gen_ran_coords(world_radius - pf_radius)
    obstacle_locations = gen_obs_coords(world_radius, pf_coords, pf_radius, obstacle_diameters)
    x_coord, y_coord = pf_coords

    # Create world and its start position
    world  = environment.world(
                T=path_len,
                world_radius=world_radius, 
                num_directions=num_dirs,
                num_sound_waves=num_waves, 
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
        angles = world.generate_waves(length=60, verbose=True)

    # Plot path
    world.plotpath()