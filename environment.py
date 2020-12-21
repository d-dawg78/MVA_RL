"""
Note: many thanks to Professor Blake Richards for the basic watermaze world implementation.

I adjusted the world for the purposes of this project, but the code skeleton was very useful!
"""
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
import mpl_toolkits.mplot3d.art3d as art3d

class world(object):

    def __init__(self, world_radius=60, platform_radius=10, platform_location=np.array([25, 25]),
                stepsize=5.0, momentum=0.2, T=60, obstacle_locations=np.array([]), obstacle_diameters=np.array([])):
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

        # Initialize dynamic variables
        self.position   = np.zeros((2, T))
        self.t          = 0
        self.prevdir    = np.zeros((2, ))


    def move(self, A):
        """
        Update agent's position using A.
        -- A: value between 0 and 7 that specifies direction from dictionary
        """

        # Check for direction out of bounds
        if (A not in np.arange(8)):
            print("Error: A must be in range 0-7.")

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
        -- A: value between 0 and 7 that specifies direction from dictionary
        """

        # Check for direction out of bounds
        if (A not in np.arange(8)):
            print("Error: A must be in range 0-7.")

        # Determine movement direction vector
        angle           = self.direction(A)
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


    def poolreflect(self, newposition):
        """
        Returns point in space if agent bumps into outside wall.
        -- newposition: agent's movement position
        """

        check, idx = self.check_for_obstacles(newposition)

        # Determine if the new position is outside the pool or within obstacle
        if (np.linalg.norm(newposition) < self.radius and check == True):
            refposition     = newposition
            refdirection    = newposition - self.position[:, self.t]

        elif (np.linalg.norm(newposition) >= self.radius):

            # Determine where the agent will hit the wall
            px = self.intercept(newposition)

            # Get the tangent vector to this point by rotating -pi / 2
            tx = np.asarray(np.matmul([[0, 1], [-1, 0]], px))

            # Get the vector of the direction of movement and tengent vector
            dx = px - self.position[:, self.t]

            # Get angle between direction of movement and tangent vector
            theta = np.arccos(np.matmul((np.divide(tx, np.linalg.norm(tx))).transpose(), np.divide(dx, np.linalg.norm(dx)))).item()

            # Rotate the remaining direction of movement vector by 2 * (pi - theta) to get the reflected direction
            ra = 2 * (np.pi - theta)
            refdirection = np.asarray(np.matmul([[np.cos(ra), -np.sin(ra)], [np.sin(ra), np.cos(ra)]], (newposition - px)))

            # Get the reflected direction
            refposition = px + refdirection

        else:

            # Determine where the agent will hit the obstacle
            px = self.obstacle_intersect(newposition, self.position[:, self.t], idx)

            # Get the tangent vector to this point by rotating -pi / 2
            tx = np.asarray(np.matmul([[0, 1], [-1, 0]], px))

            # Get the vector of the direction of movement and tengent vector
            dx = px - self.position[:, self.t]

            # Get angle between direction of movement and tangent vector
            theta = np.arccos(np.matmul((np.divide(tx, np.linalg.norm(tx))).transpose(), np.divide(dx, np.linalg.norm(dx)))).item()

            # Rotate the remaining direction of movement vector by 2 * (pi - theta) to get the reflected direction
            ra = 2 * (np.pi - theta)
            refdirection = np.asarray(np.matmul([[np.cos(ra), -np.sin(ra)], [np.sin(ra), np.cos(ra)]], (newposition - px)))

            # Get the reflected direction
            refposition = px + refdirection

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

        return [refposition, refdirection]


    def intercept(self, newposition):
        """
        Function that checks when and where the agent hits the edge of the pool or obtacle.
        Returns point in space where agent will intercept with world wall or obtacle.
        -- newposition: agent's movement position
        """

        p1 = self.position[:, self.t]
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
        plt.plot(self.position[0, 0:self.t], self.position[1, 0:self.t], color='k', ls='dotted')

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