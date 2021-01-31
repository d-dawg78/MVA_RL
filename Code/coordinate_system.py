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
        -- g: discounting factor
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
        -- lr: learning rate
        -- num_directions: number of directions actor can use.
        -- num_place_cells: total number of place cells.
        """

        self.W   = np.zeros([num_directions + 1, num_place_cells])
        self.C   = np.zeros(num_place_cells)
        self.lr  = lr
        self.nd  = num_directions

    def probs(self, place_cells):
        """
        Compute probabilities.
        -- place_cells: array of place cells.
        """

        currs = np.zeros(len(place_cells))

        for x in range(len(place_cells)):
            currs[x] = place_cells[x].current

        # Update coordinate vector before computing probabilities
        self.W[self.nd, :] = self.C[:]

        a = np.dot(self.W, currs)

        # Overflow avoidance
        max_a = np.max(a)
        num = 2 * (a - max_a)
        num = np.exp(num)

        den   = num.sum()

        p = num / den

        return p

    def coord_update(self, error):
        """
        Update coordinate system when its move was chosen.
        -- error: error computed at each step.
        """

        self.C[:] += self.lr * error

    def weight_update(self, direction, error, place_cells):
        """
        Weight update function. 
        -- direction: direction chosen at current step
        -- error: error computed at each step
        -- place_cells: array of place cells at each step
        """

        # Only update weights when coordinate system has not been used.
        if (direction == self.nd):
            return

        prevs = np.zeros(len(place_cells))

        for x in range(len(place_cells)):
            prevs[x] = place_cells[x].prev

        self.W[direction, :] += self.lr * error * prevs[:]

        return


class Coord_System(object):
    """
    Class for updating X and Y weights in coordinate system.
    """

    def __init__(self, lr, lam, num_place_cells):
        """
        Arguments:
        -- num_place_cells: number of place cells.
        """

        self.X   = np.zeros(num_place_cells, dtype=np.float64)
        self.Y   = np.zeros(num_place_cells, dtype=np.float64)
        self.lr  = lr
        self.lam = lam
        self.fps = []

    def probs(self, place_cells):
        """
        Compute probabilities.
        -- place_cells: array of place cells.
        """

        prevs = np.zeros(len(place_cells))
        currs = np.zeros(len(place_cells))

        for x in range(len(place_cells)):
            prevs[:] = place_cells[x].prev
            currs[:] = place_cells[x].current

        X_currs = np.dot(currs, self.X)
        X_prevs = np.dot(prevs, self.X)

        Y_currs = np.dot(currs, self.Y)
        Y_prevs = np.dot(prevs, self.Y)

        return X_currs, X_prevs, Y_currs, Y_prevs

    def weight_update(self, place_cells, delta_X, delta_Y, X_currs, X_prevs, Y_currs, Y_prevs):
        """
        Function for updating X and Y weights.
        -- delta_X: difference between current and previous position (X coordinate)
        -- delta_Y: difference between current and previous position (Y coordinate)
        -- X_currs: cuurent position probabilities (X coordinate)
        -- Y_currs: current position probabilities (Y coordinate)
        -- X_prevs: previous position probabilities (X coordinate)
        -- Y_prevs: previous position probabilities (Y coordinate)
        """

        prevs = np.zeros(len(place_cells))

        for x in range(len(place_cells)):
            prevs[x] = place_cells[x].prev

        summation = np.zeros(len(place_cells), dtype=np.float64)

        t = len(self.fps)
        k = 1

        for el in self.fps:
            #for x in range(len(el)):
            summation[:] += self.lam**(t - k) * el[:]
            k += 1

        self.X[:] += self.lr * (delta_X - (X_currs - X_prevs) * summation)
        self.Y[:] += self.lr * (delta_Y - (Y_currs - Y_prevs) * summation)

        return