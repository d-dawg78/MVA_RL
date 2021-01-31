import numpy as np 
import place_cell


class Critic(object):
    """
    Class that defines the critic and its functions.
    """

    def __init__(self, lr, g, num_place_cells):
        """
        Arguments:
        -- num_place_cells: number of place cells.
        -- lr: learning rate
        -- gamma: discounting factor
        """

        self.W     = np.zeros(num_place_cells)
        self.lr    = lr
        self.gamma = g

    def weight_update(self, R, place_cells):
        """
        Weight update function. 
        -- R: reward
        -- place_cells: place cell array
        """

        prevs = np.zeros(len(place_cells))
        currs = np.zeros(len(place_cells))

        for x in range(len(place_cells)):
            prevs[x] = place_cells[x].prev
            currs[x] = place_cells[x].current

        C_currs = np.dot(currs, self.W)
        C_prevs = np.dot(prevs, self.W)

        error   = R + self.gamma * C_currs - C_prevs

        self.W += self.lr * (error * prevs)

        return error


class Actor(object):
    """
    Class that defines the actor and its functions.
    """

    def __init__(self, lr, num_directions, num_place_cells):
        """
        Arguments:
        -- num_directions: number of directions actor can use.
        -- num_place_cells: total number of place cells.
        """

        self.W  = np.zeros([num_directions, num_place_cells])
        self.lr = lr

    def probs(self, place_cells):
        """
        Compute probabilities.
        -- place_cells: array of place cells.
        """

        currs = np.zeros(len(place_cells))

        for x in range(len(place_cells)):
            currs[x] = place_cells[x].current

        a = np.dot(self.W, currs)

        # Overflow avoidance
        max_a = np.max(a)
        num = 2 * (a - max_a)
        num = np.exp(num)

        den   = num.sum()

        p = num / den

        return p

    def weight_update(self, direction, error, place_cells):
        """
        Weight update function. 
        -- direction: direction chosen at current step
        -- error: error computed at each step
        -- place_cells: array of place cells at each step
        """

        prevs = np.zeros(len(place_cells))

        for x in range(len(place_cells)):
            prevs[x] = place_cells[x].prev

        self.W[direction, :] += self.lr * error * prevs[:]

        return