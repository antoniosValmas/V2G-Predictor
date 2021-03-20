import config as cfg


class Parking:
    """
    A class representing a V2G parking facility

    ### Arguments:
        capacity (``int``) :
            description: The total parking spaces

    ### Attributes
        parking_spaces (``ParkingSpace[]``) :
            description: An array containing all available parking spaces objects
    """

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._parking_spaces = [ParkingSpace() for i in range(capacity)]

    def get_capacity(self):
        """
        Get capacity of parking

        ### Returns
            capacity (`int`): The total parking spaces
        """
        return self._capacity


class ParkingSpace:
    """
    A class representing a V2G parking space

    ### Arguments:
        max_charging_rate (``float``) :
            description: The maximum possible charging rate
            default: `config.MAX_CHARGING_RATE`
        max_discharging_rate (``float``) :
            description: The maximum possible discharging rate
            default: `config.MAX_DISCHARGING_RATE`
    """

    def __init__(
        self,
        max_charging_rate: float = cfg.MAX_CHARGING_RATE,
        max_discharging_rate: float = cfg.MAX_DISCHARGING_RATE,
    ):
        self._max_charging_rate = max_charging_rate
        self._max_discharging_rate = max_discharging_rate

    def get_max_charging_rate(self):
        """
        Get the maximum possible charging rate

        ### Returns:
            float : The maximum possible charging rate
        """
        return self._max_charging_rate

    def get_max_discharging_rate(self):
        """
        The maximum possible discharging rate

        ### Returns:
            float : The maximum possible discharging rate
        """
        return self._max_discharging_rate
