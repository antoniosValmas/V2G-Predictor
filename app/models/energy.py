import numpy as np


class EnergyCurve:
    """
    A class that holds the energy curve along with its derivatives

    ### Arguments:
        data (`Tuple(datetime, float)[]`) :
            description: The raw data array. Each point consists of the timestamp and the cost of energy
    """

    def __init__(self, data):
        self._data = data

        self._x = [x for (x, _) in data]
        self._y = [y for (_, y) in data]
        self._first_derivate = np.gradient(self._y)
        self._second_derivate = np.gradient(self._first_derivate)

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def get_first_derivate(self):
        return self._first_derivate

    def get_second_derivate(self):
        return self._second_derivate
