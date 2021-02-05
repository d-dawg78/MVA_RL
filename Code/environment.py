import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
import mpl_toolkits.mplot3d.art3d as art3d


class world(object):
    """
    Class for world creation. Edited watermaze.
    """

    def __init__(self, world_radius=60, platform_radius=10, platform_location=np.array([25, 25]),
                stepsize=5.0, momentum=0.2, T=60, obstacle_locations=np.array([]), obstacle_diameters=np.array([]),
                num_sound_waves=1, agent_radius=2, num_directions=8):
        """
        Arguments:
        -- world_radius: radius of world
        -- platform_radius: platform radius
        -- platform_location: location of platform center
        -- stepsize: how far agent moves in one step
        -- momentum: ratio of old / new movement 
        -- T: maximum trial time
        -- obstacle_locations: xy positions for obstacles (smallest square values)
        -- obstacle_diameters: square widths
        -- num_sound_waves: number of echolocating waves
        -- agent_radius: agent's radius (to use with current position)
        """
        
        # Initialize all world parameters
        self.radius                 = world_radius
        self.platform_radius        = platform_radius
        self.platform_location      = platform_location
        self.stepsize               = stepsize
        self.momentum               = momentum
        self.T                      = T
        self.obstacle_locations     = obstacle_locations
        self.obstacle_diameters     = obstacle_diameters
        self.num_sound_waves        = num_sound_waves
        self.agent_radius           = agent_radius
        self.num_directions         = num_directions
        self.colisions              = 0

        if (num_directions == 8):
            # Direction dictionary
            self.direction = {
                0: np.pi / 2,       # north
                1: np.pi / 4,       # north-east
                2: 0,               # east
                3: 7 * np.pi / 4,   # south-east
                4: 3 * np.pi / 2,   # south
                5: 5 * np.pi / 4,   # south-west
                6: np.pi,           # west
                7: 3 * np.pi / 4,   # north-west
            }

        elif (num_directions == 16):
            # Direction dictionary
            self.direction = {
                0: np.pi / 2,       # north
                1: 3 * np.pi / 8,   # between north and north-east
                2: np.pi / 4,       # north-east
                3: np.pi / 8,       # between north-east and east
                4: 0,               # east
                5: 15 * np.pi / 8,  # between east and south-east
                6: 7 * np.pi / 4,   # south-east
                7: 13 * np.pi / 8,  # between south-east and south
                8: 3 * np.pi / 2,   # south
                9: 11 * np.pi / 8,  # between south and south-west
                10: 5 * np.pi / 4,  # south-west
                11: 9 * np.pi / 8,  # between south-west and west
                12: np.pi,          # west
                13: 7 * np.pi / 8,  # between west and noth-west
                14: 3 * np.pi / 4,  # north-west
                15: 5 * np.pi / 8,  # between noth-west and north
            }

        else:
            raise Exception("Error: number of directions can only be 8 or 16 !")

        # Initialize dynamic variables
        self.position   = np.zeros((2, T))
        self.t          = 0
        self.prevdir    = np.zeros((2, ))


    def move(self, A):
        """
        Update agent's position using A.
        -- A: value between 0 and num_directions that specifies direction from dictionary
        """

        # Check for direction out of bounds
        #if (A not in np.arange(self.num_directions)):
        #    print("Error: A must be in range 0-{}.".format(self.num_directions))

        # Determine movement direction vector
        angle           = self.direction[A]
        newdirection    = np.array([np.cos(angle), np.sin(angle)])

        # Add in momentum to reflect movement dynamics
        direction = (1.0 - self.momentum) * newdirection + self.momentum * self.prevdir
        direction = direction / np.sqrt((direction**2).sum())
        direction = direction * self.stepsize

        # Update the position (note that agent bounces of wall and objects)
        [newposition, direction] = self.poolreflect(self.position[:, self.t] + direction)

        # When agent is at the edge of the pool, move it in
        if (np.linalg.norm(newposition) == self.radius):
            newposition = np.multiply(np.divide(newposition, np.linalg.norm(newposition)), self.radius - 1)

        # Update position, time, and previous direction
        self.position[:, self.t + 1]    = newposition
        self.t                          = self.t + 1
        self.prevdir                    = direction


    def future_position(self, A):
        """
        Future position based on supposed movement. No positional update
        -- A: value between 0 and num_directions that specifies direction from dictionary
        """

        # Check for direction out of bounds
        #if (A not in np.arange(8)):
        #    print("Error: A must be in range 0-7.")

        # Determine movement direction vector
        angle           = self.direction[A]
        newdirection    = np.array([np.cos(angle), np.sin(angle)])

        # Add in momentum to reflect movement dynamics
        direction = (1.0 - self.momentum) * newdirection + self.momentum * self.prevdir
        direction = direction / np.sqrt((direction**2).sum())
        direction = direction * self.stepsize

        # Update the position (note that agent bounces of wall and objects)
        [newposition, direction] = self.poolreflect(self.position[:, self.t] + direction, future=True)

        # When agent is at the edge of the pool, move it in
        if (np.linalg.norm(newposition) == self.radius):
            newposition = np.multiply(np.divide(newposition, np.linalg.norm(newposition)), self.radius - 1)

        return newposition


    def check_for_obstacles(self, newposition):
        """
        Determine whether agent will hit an obstacle.
        -- newposition: position where agent wants to move
        """

        for x in range(len(self.obstacle_locations)):
            if (newposition[0] >= self.obstacle_locations[x][0] and newposition[0] <= self.obstacle_locations[x][0] + self.obstacle_diameters[x] and 
                newposition[1] >= self.obstacle_locations[x][1] and newposition[1] <= self.obstacle_locations[x][1] + self.obstacle_diameters[x]):
                return False, x
            
            else:
                continue

        return True, -1


    def obstacle_intersect(self, newposition, old_position, idx):
        """
        Function for finding the intersection point between the obstacle and new position.
        -- newposition: point to which agent wants to move
        -- old_position: point at which agent currently is
        -- idx: obstacle index
        """

        # Get obstacle location and diameter
        obstacle_loc    = self.obstacle_locations[idx]
        obstacle_diam   = self.obstacle_diameters[idx]

        # Get line functions
        x1 = obstacle_loc[0]
        x2 = obstacle_loc[0] + obstacle_diam
        y1 = obstacle_loc[1]
        y2 = obstacle_loc[1] + obstacle_diam

        # Determine newposition / oldposition line parameter
        m = (old_position[1] - newposition[1]) / (old_position[0] - newposition[0])

        # Find intersection with obtacle boundaries
        x_top = ((y2 - newposition[1]) / m) + newposition[0]
        x_bot = ((y1 - newposition[1]) / m) + newposition[0]
        y_lft = m * (x1 - newposition[0]) + newposition[1]
        y_rgt = m * (x2 - newposition[0]) + newposition[1]

        # Find closest intersection point
        list_of_intersections = [[x_top, y2], [x_bot, y1], [x1, y_lft], [x2, y_rgt]]

        dist = math.inf
        intersection_point = [0, 0]

        for el in list_of_intersections:
            temp        = np.asarray(el)
            temp_dist   = np.linalg.norm(newposition - temp)

            if (temp_dist < dist):
                dist = temp_dist
                intersection_point = el

        intersection_point = np.asarray(intersection_point)

        return intersection_point


    def poolreflect(self, newposition, prev_pos=np.array([]), wave=False, future=False, st=-1):
        """
        Returns point in space if agent bumps into outside wall.
        -- newposition: agent or wave's position to test
        -- prev_pos: previous position; if not set: poolreflect deals with agent, else with sound wave.
        """

        status = st

        # Set to last position when agent
        if (len(prev_pos) == 0):
            prev_pos = self.position[:, self.t]

        check, idx = self.check_for_obstacles(newposition)

        # Determine if the new position is outside the pool or within obstacle
        if (np.linalg.norm(newposition) < self.radius and check == True):
            refposition     = newposition
            refdirection    = newposition - prev_pos

        elif (np.linalg.norm(newposition) >= self.radius):

            if (wave == False and future == False):
                self.colisions += 1

            # Determine where the agent will hit the wall
            px = self.intercept(newposition, prev_pos)

            # Get the tangent vector to this point by rotating -pi / 2
            tx = np.asarray(np.matmul([[0, 1], [-1, 0]], px))

            # Get the vector of the direction of movement and tengent vector
            dx = px - prev_pos

            # Get angle between direction of movement and tangent vector
            theta = np.arccos(np.matmul((np.divide(tx, np.linalg.norm(tx))).transpose(), np.divide(dx, np.linalg.norm(dx)))).item()

            # Rotate the remaining direction of movement vector by 2 * (pi - theta) to get the reflected direction
            ra = 2 * (np.pi - theta)
            refdirection = np.asarray(np.matmul([[np.cos(ra), -np.sin(ra)], [np.sin(ra), np.cos(ra)]], (newposition - px)))

            # Get the reflected direction
            refposition = px + refdirection

            status = 1

        else:

            if (wave == False and future == False):
                self.colisions += 1

            # Determine where the agent will hit the obstacle
            px = self.obstacle_intersect(newposition, prev_pos, idx)

            # Get the tangent vector to this point by rotating -pi / 2
            tx = np.asarray(np.matmul([[0, 1], [-1, 0]], px))

            # Get the vector of the direction of movement and tengent vector
            dx = px - prev_pos

            # Get angle between direction of movement and tangent vector
            theta = np.arccos(np.matmul((np.divide(tx, np.linalg.norm(tx))).transpose(), np.divide(dx, np.linalg.norm(dx)))).item()

            # Rotate the remaining direction of movement vector by 2 * (pi - theta) to get the reflected direction
            ra = 2 * (np.pi - theta)
            refdirection = np.asarray(np.matmul([[np.cos(ra), -np.sin(ra)], [np.sin(ra), np.cos(ra)]], (newposition - px)))

            # Get the reflected direction
            refposition = px + refdirection

            status = 1

        # Make sure new position is inside the pool
        if (np.linalg.norm(refposition) > self.radius):
            refposition = np.multiply(refposition / np.linalg.norm(refposition), self.radius - 1)

        # Make sure new position is not inside obstacle
        check, idx = self.check_for_obstacles(refposition)

        if (check == False):
            
            # Check x-value
            if (refposition[0] >= self.obstacle_locations[idx][0] and refposition[0] <= self.obstacle_locations[idx][0] + self.obstacle_diameters[idx]):
                dist1 = abs(refposition[0] - self.obstacle_locations[idx][0])
                dist2 = abs(refposition[0] - (self.obstacle_locations[idx][0] + self.obstacle_diameters[idx]))

                if (dist1 <= dist2):
                    refposition[0] = self.obstacle_locations[idx][0] - 1

                else:
                    refposition[0] = self.obstacle_locations[idx][0] + self.obstacle_diameters[idx] + 1

            # Check y-value
            if (refposition[1] >= self.obstacle_locations[idx][1] and refposition[1] <= self.obstacle_locations[idx][1] + self.obstacle_diameters[idx]):
                dist1 = abs(refposition[1] - self.obstacle_locations[idx][1])
                dist2 = abs(refposition[1] - (self.obstacle_locations[idx][1] + self.obstacle_diameters[idx]))

                if (dist1 <= dist2):
                    refposition[1] = self.obstacle_locations[idx][1] - 1

                else:
                    refposition[1] = self.obstacle_locations[idx][1] + self.obstacle_diameters[idx] + 1

        if (wave == True):
            return (status, [refposition, refdirection])

        else:
            return [refposition, refdirection]


    # NOTE: Code from here https://stackoverflow.com/questions/30844482/what-is-most-efficient-way-to-find-the-intersection-of-a-line-and-a-circle-in-py
    def circle_line_segment_intersection(self, circle_center, circle_radius, pt1, pt2, full_line=True, tangent_tol=1e-9):
        """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

        :param circle_center: The (x, y) location of the circle center
        :param circle_radius: The radius of the circle
        :param pt1: The (x, y) location of the first point of the segment
        :param pt2: The (x, y) location of the second point of the segment
        :param full_line: True to find intersections along full line - not just in the segment.  False will just return intersections within the segment.
        :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
        :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.

        Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
        """

        (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
        (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
        dx, dy = (x2 - x1), (y2 - y1)
        dr = (dx ** 2 + dy ** 2)**.5
        big_d = x1 * y2 - x2 * y1
        discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

        if discriminant < 0:  # No intersection between circle and line
            return []
            #print("No intersection between circle and line")
        else:  # There may be 0, 1, or 2 intersections with the segment
            #print("Should be good.")
            intersections = [
                [cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
                cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2]
                for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
            if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
                fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
                intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
            if len(intersections) == 2 and abs(discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
                return np.asarray(intersections[0])
            else:
                distance    = math.inf
                point       = np.asarray([0, 0])

                for pt in intersections:
                    temp        = np.asarray(pt)
                    temp_dist   = np.linalg.norm(pt1 - temp)

                    if (temp_dist < distance):
                        distance = temp_dist
                        point    = temp

                return point

    
    def platform_reflect(self, newposition, prev_pos, st):
        """
        Function for reflecting sound wave across platform.
        """

        status = st

        if (np.sqrt(np.sum((newposition - self.platform_location)**2)) <= (self.platform_radius + 1)):

            # Find where agent crosses to platform
            px = self.circle_line_segment_intersection(self.platform_location, self.platform_radius, newposition, prev_pos)

            # Treat when crossing position found is not correct
            if (px == [] or np.isnan(px).any() == True):
                return status, np.asarray([float('NaN'), float('NaN')])

            # Set status to 2 (hit platform)
            status = 2

            # Get the tangent vector to this point by rotating -pi / 2
            tx = np.asarray(np.matmul([[0, 1], [-1, 0]], px))

            # Get the vector of the direction of movement and tangent vector
            dx = px - prev_pos
            
            """
            print(newposition)
            print(prev_pos)
            print(px)
            print(dx)
            """

            # Nudge dx if it equals 0 (crossing and previous location are equal)
            if (dx[0] == 0.):
                dx[0] = 0.001

            if (dx[1] == 0.):
                dx[1] = 0.001

            # Get angle between direction of movement and tangent vector
            theta = np.arccos(np.matmul((np.divide(tx, np.linalg.norm(tx))).transpose(), np.divide(dx, np.linalg.norm(dx)))).item()

            # Rotate the remaining direction of movement vector by 2 * (pi - theta) to get the reflected direction
            ra = 2 * (np.pi - theta)
            refdirection = np.asarray(np.matmul([[np.cos(ra), -np.sin(ra)], [np.sin(ra), np.cos(ra)]], (newposition - px)))

            # Get the reflected direction
            refposition = px + refdirection

            # Make sure new location is not in platform location
            if (np.sqrt(np.sum((refposition - self.platform_location)**2)) < (self.platform_radius + 1)):

                check_pos = self.circle_line_segment_intersection(self.platform_location, self.platform_radius, refposition, newposition)

                if (check_pos == [] or np.isnan(px).any() == True):
                    #print("BUG: reflected location on platform but no new position correction.")
                    return status, np.asarray([refposition, refdirection])
                
                else:
                    return status, np.asarray([check_pos, refdirection])

            return status, np.asarray([refposition, refdirection])

        return status, np.asarray([float('NaN'), float('NaN')])


    def intercept(self, newposition, prev_pos):
        """
        Function that checks when and where the agent hits the edge of the pool or obtacle.
        Returns point in space where agent will intercept with world wall or obtacle.
        -- newposition: agent's movement position
        """
        p1 = prev_pos
        p2 = newposition

        # Calculate terms used to find the point of intersection
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dr = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
        D  = p1[0] * p2[1] - p2[0] * p1[1]
        sy = np.sign(dy)

        if (sy == 0):
            sy = 1.0

        # Calculate the potential points of intercection
        pp1 = np.zeros((2, ))
        pp2 = np.zeros((2, ))

        pp1[0] = (D * dy + sy * dx * np.sqrt(np.power(self.radius, 2) * np.power(dr, 2) - np.power(D, 2))) / np.power(dr, 2)
        pp2[0] = (D * dy - sy * dx * np.sqrt(np.power(self.radius, 2) * np.power(dr, 2) - np.power(D, 2))) / np.power(dr, 2)
        pp1[1] = (-D * dx + np.absolute(dy) * np.sqrt(np.power(self.radius, 2) * np.power(dr, 2) - np.power(D, 2))) / np.power(dr, 2)
        pp2[1] = (-D * dx - np.absolute(dy) * np.sqrt(np.power(self.radius, 2) * np.power(dr, 2) - np.power(D, 2))) / np.power(dr, 2)

        # Determine appropriate intersection point
        if (np.linalg.norm(p2 - pp1) < np.linalg.norm(p2 - pp2)):
            px = pp1

        else:
            px = pp2

        return px


    def startposition(self):
        """
        Set agent start position along one of the main cardinal axes.
        Calculate vector angle.
        """

        condition   = 2 * np.random.randint(0, 4)
        angle       = self.direction[condition]

        self.position[:, 0] = np.asarray([np.cos(angle), np.sin(angle)]) * (self.radius - 1)


    def plotpath(self):
        """
        Plot the agent's path through world.
        """

        fig = plt.figure()
        ax  = fig.gca()

        # Create pool perimeter
        pool_perimeter = plt.Circle((0, 0), self.radius, fill=False, color='b', ls='-')
        ax.add_artist(pool_perimeter)

        # Create the platform
        platform = plt.Circle(self.platform_location, self.platform_radius, fill=True, color='g', ls='-')
        ax.add_artist(platform)

        # Create the obstacles
        for x in range(len(self.obstacle_locations)):
            obstacle = plt.Rectangle(self.obstacle_locations[x], self.obstacle_diameters[x], self.obstacle_diameters[x], fill=True, color='r', ls='-')
            ax.add_artist(obstacle)

        # Plot the path
        #plt.plot(self.position[0, 0:self.t], self.position[1, 0:self.t], color='k', ls='dotted')
        plt.plot(self.position[0, 0:self.t], self.position[1, 0:self.t], color='k')

        # Plot the final location and starting location
        plt.plot(self.position[0, 0], self.position[1, 0], color='b', marker='o', markersize=4, markerfacecolor='b')
        plt.plot(self.position[0, self.t - 1], self.position[1, self.t - 1], color='r', marker='o', markersize=6, markerfacecolor='r')

        # Adjust the axis
        ax.axis('equal')
        ax.set_xlim((-self.radius - 50, self.radius + 50))
        ax.set_ylim((-self.radius - 50, self.radius + 50))
        plt.xticks(np.arange(-self.radius, self.radius + 20, step=20))
        plt.yticks(np.arange(-self.radius, self.radius + 20, step=20))
        ax.set_xlabel('X Position (cm)')
        ax.set_ylabel('Y position (cm)')

        # Turn on the grid
        plt.grid(True)
        plt.tight_layout()

        # Show the figure
        plt.show()


    def timeup(self):
        """
        True if time trial is over, false otherwise.
        """

        return self.t > (self.T - 2)


    def atgoal(self):
        """
        True if agent is on the platform, false otherwise.
        """

        return np.sqrt(np.sum((self.position[:, self.t] - self.platform_location)**2)) <= (self.platform_radius + 1)        


    def move_sound_wave(self, pos, angle, max_travel):
        """
        Move sound wave given position and angle.
        """
        index           = 0
        prev_wave_dir   = self.prevdir
        position        = pos
        all_positions   = [position]
        status          = 0

        while (index < max_travel):

            # Determine movement direction vector
            newdirection    = np.array([np.cos(angle), np.sin(angle)])

            # Add in momentum to reflect movement dynamics
            direction = (1.0 - self.momentum) * newdirection + self.momentum * prev_wave_dir
            direction = direction / np.sqrt((direction**2).sum())
            direction = direction * self.stepsize

            # Update the position (note that agent bounces of wall and objects)
            status, [newposition, direction] = self.poolreflect(position + direction, position, wave=True, st=status)
            
            temp_status, platresults = self.platform_reflect(newposition, position, status)
            #try:
            if (np.isnan(platresults.flatten()).any() == False):
                newposition = platresults[0]
                direction   = platresults[1]
                status      = temp_status
            #except:
            #    print(platresults)
            #raise

            # When agent is at the edge of the pool, move it in
            if (np.linalg.norm(newposition) == self.radius):
                newposition = np.multiply(np.divide(newposition, np.linalg.norm(newposition)), self.radius - 1)

            # Update position, time, and previous direction
            all_positions.append(newposition)

            # Check if sound wave hits agent
            if (np.sqrt(np.sum((newposition - pos)**2)) <= (self.agent_radius + 1)):
                angle = (np.arctan2(-direction[1], -direction[0]) + 2 * np.pi) % (2 * np.pi)
                #print("Hit agent: status {} and direction {}".format(status, angle))
                return status, angle, all_positions


            prev_wave_dir   = direction
            position        = newposition
            angle           = np.arctan2(prev_wave_dir[1], prev_wave_dir[0])
            index           = index + 1

        #print(status)

        return 0, angle, all_positions


    def generate_waves(self, length, verbose=False):
        """
        Function for generating waves fron location for ecolocation.
        """   

        # Get current position
        position    = self.position[:, self.t]
        dir_vector  = self.prevdir
        dir_angle   = np.arctan2(dir_vector[1], dir_vector[0])

        # Generate directions
        low_bound   = dir_angle - (np.pi / 4)
        upp_bound   = dir_angle + (np.pi / 4)

        #print("Lower bound: {}. Middle: {}. Upper bound: {}".format(low_bound, dir_angle, upp_bound))

        if (self.num_sound_waves == 0):
            angles  = {}
        
        else:
            # Split waves in equal directions; dictionary contains angle and status
            angles  = np.linspace(low_bound, upp_bound, self.num_sound_waves)
            paths   = np.linspace(low_bound, upp_bound, self.num_sound_waves)
            angles  = dict((el, 0) for el in angles)
            paths   = dict((el, 0) for el in paths)

            #print(angles)

            # Make sound wave travel
            for key in angles:
                status, direction, all_positions = self.move_sound_wave(position, key, length)
                all_positions = np.asarray(all_positions)
                paths[key]  = all_positions
                angles[key] = (status, direction)

            if (verbose == True):

                fig = plt.figure()
                ax  = fig.gca()

                # Create pool perimeter
                pool_perimeter = plt.Circle((0, 0), self.radius, fill=False, color='b', ls='-')
                ax.add_artist(pool_perimeter)

                # Create the platform
                platform = plt.Circle(self.platform_location, self.platform_radius, fill=True, color='g', ls='-')
                ax.add_artist(platform)

                # Create the obstacles
                for x in range(len(self.obstacle_locations)):
                    obstacle = plt.Rectangle(self.obstacle_locations[x], self.obstacle_diameters[x], self.obstacle_diameters[x], fill=True, color='r', ls='-')
                    ax.add_artist(obstacle)

                # Plot sound waves
                for key in paths:
                    plt.plot(paths[key][:, 0], paths[key][:, 1], color='r')

                # Plot the path
                #plt.plot(self.position[0, 0:self.t], self.position[1, 0:self.t], color='k', ls='dotted')
                plt.plot(self.position[0, 0:self.t], self.position[1, 0:self.t], color='k')

                # Plot the final location and starting location
                plt.plot(self.position[0, 0], self.position[1, 0], color='b', marker='o', markersize=4, markerfacecolor='b')
                plt.plot(self.position[0, self.t - 1], self.position[1, self.t - 1], color='r', marker='o', markersize=6, markerfacecolor='r')

                # Adjust the axis
                ax.axis('equal')
                ax.set_xlim((-self.radius - 50, self.radius + 50))
                ax.set_ylim((-self.radius - 50, self.radius + 50))
                plt.xticks(np.arange(-self.radius, self.radius + 20, step=20))
                plt.yticks(np.arange(-self.radius, self.radius + 20, step=20))
                ax.set_xlabel('X Position (cm)')
                ax.set_ylabel('Y position (cm)')

                # Turn on the grid
                plt.grid(True)
                plt.tight_layout()

                # Show the figure
                plt.show()

        return angles