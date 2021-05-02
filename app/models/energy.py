from typing import List, Tuple
import datetime


class EnergyCurve:
    """
    A class that holds the energy curve along with its derivatives

    ### Arguments:
        data (`Tuple(datetime, float)[]`) :
            description: The raw data array. Each point consists of the timestamp and the cost of energy
    """
    _data: List[Tuple[datetime.datetime, float]] = []
    _x: List[datetime.datetime]
    _y: List[float]
    _start: int = 0
    _end: int = 24

    def __init__(self, dataFile: str):
        self._data = []
        self._x = []
        self._y = []
        with open(dataFile) as csv:
            for line in csv.readlines():
                date, value = line.split(',')
                date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
                value = float(value)
                self._data.append((date, value))
                self._x.append(date)
                self._y.append(value)

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

    def get_next_batch(self, normalized=True):
        """
        Returns the next batch of values
        Moves window to the next values

        ### Returns
            float[24] : A 24 size array with the energy cost
            bool : Whether we reached the end of the data
        """
        ret = [val / (100 if normalized else 1) for val in self._y[self._start:self._end]]
        if len(self._data) > self._end:
            self._start += 1
            self._end += 1
        else:
            print('Done')
        return (ret, len(self._data) == self._end)

    def reset(self):
        self._start = 0
        self._end = 24

    def get_current_cost(self):
        """
        Get current energy cost
        """
        return self._y[self._start - 1]
