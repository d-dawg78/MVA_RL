import numpy as np
import math
import matplotlib.pyplot as plt

class PlaceCell(object):
    """
    Class for mimicking place cell activity.
    """

    def __init__(self, center, std):
        """
        Arguments:
        -- center: place cell coordinate center.
        -- std: place cell standard deviation.

        current and prev denote current and previous activations.
        """

        self.center   = center
        self.std      = std
        self. current = 0
        self.prev     = 0


    def activate(self, pos):
        """
        Activation function from equation 1.
        -- pos: agent position.
        """

        self.prev = self.current

        num = np.linalg.norm(pos - self.center)
        num = -np.square(num)

        den = 2 * self.std * self.std

        f_p = np.exp(num / den)

        self.current = f_p

        return


# Code for evenly distributing Place Cell centers in circular world from here:
# https://stackoverflow.com/questions/28567166/uniformly-distribute-x-points-inside-a-circle

def fun_radius(k, n, b):
    if (k > n - b):
        r = 1

    else:
        r = math.sqrt(k - 1/2) / math.sqrt(n - (b + 1) / 2)

    return r


def place_cells(N, std, radius, verbose=False):
    """
    Function for equally distributing world place cell centers in watermaze.

    Arguments:
    -- N: number of place cells
    -- std: standard deviation value for each place cell
    -- radius: world radius
    -- verbose: plot place cell distribution or not
    """

    arr = np.full(N, fill_value=PlaceCell([0, 0], std))

    # Number of boundary points and golden ratio
    b   = round(math.sqrt(N))
    phi = (math.sqrt(5) + 1) / 2

    for k in range(1, N + 1, 1):
        r     = fun_radius(k, N, b)
        theta = 2 * math.pi * k / (phi * phi)
        arr[k - 1] = PlaceCell(np.array([r * math.cos(theta) * radius, r * math.sin(theta) * radius]), std)

    if (verbose == True):
    
        for l in range(len(arr)):
            plt.plot(arr[l].center[0], arr[l].center[1], 'ro')
    
        plt.title('Place Cell Centers')
        plt.show()

    return arr