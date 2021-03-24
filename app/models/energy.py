from typing import List, Tuple
import numpy as np
import datetime


class EnergyCurve:
    """
    A class that holds the energy curve along with its derivatives

    ### Arguments:
        data (`Tuple(datetime, float)[]`) :
            description: The raw data array. Each point consists of the timestamp and the cost of energy
    """

    def __init__(self, data: List[Tuple[datetime.datetime, float]]):
        self._data = data

        self._x = [x for (x, _) in data]
        self._y = [y for (_, y) in data]
        self._first_derivate = np.gradient(self._y)
        self._second_derivate = np.gradient(self._first_derivate)

    def get_raw_data(self):
        """
        Get the raw energy data

        ### Returns
            Tuple(datetime, float)[] : The raw energy data
        """
        return self._data

    def get_x(self):
        """
        Get the x values (datetimes)

        ### Returns
            datetime[] : The x values of the data
        """
        return self._x

    def get_y(self):
        """
        Get the energy cost values throughout time

        ### Returns:
            float[] : The y values of the data
        """
        return self._y

    def get_first_derivate(self):
        """
        Get the first derivative of energy demand curve

        ### Returns:
            float[] : Thw first derivative values of the data
        """
        return self._first_derivate

    def get_second_derivate(self):
        """
        Get the second derivative of energy demand curve

        ### Returns:
            float[] : Thw second derivative values of the data
        . :math:d/23
        """
        return self._second_derivate
