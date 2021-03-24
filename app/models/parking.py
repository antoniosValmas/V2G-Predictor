import random
import json
from typing import Any, Dict
import config as cfg
from app.error_handling import ParkingIsFull
from app.models.vehicle import Vehicle


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

    ### Attributes:
        _vehicle (``Vehicle``) :
            description: The assigned vehicle
            default: None
    """

    def __init__(
        self,
        max_charging_rate: float = cfg.MAX_CHARGING_RATE,
        max_discharging_rate: float = cfg.MAX_DISCHARGING_RATE,
    ):
        self._max_charging_rate = max_charging_rate
        self._max_discharging_rate = max_discharging_rate
        self._vehicle = None

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

    def is_free(self):
        return self._vehicle is None

    def assign_vehicle(self, vehicle: Vehicle):
        """
        Assign a new vehicle to the parking space

        ### Arguments:
            vehicle (``Vehicle``) :
                description: A non-parked instance of the vehicle class
        """
        vehicle.park(
            self._max_charging_rate,
            self._max_discharging_rate,
        )
        self._vehicle = vehicle

    def remove_vehicle(self):
        """
        Removes stored vehicle instance
        """
        self._vehicle = None

    def toJson(self) -> Dict[str, Any]:
        return {
            "class": ParkingSpace.__name__,
            "max_charging_rate": self._max_charging_rate,
            "max_discharging_rate": self._max_discharging_rate,
            "free": self.is_free(),
            "vehicle": self._vehicle and self._vehicle.toJson(),
        }

    def __repr__(self) -> str:
        return json.dumps(self.toJson(), indent=4)


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
        self._parking_spaces = [ParkingSpace() for _ in range(capacity)]

    def get_capacity(self):
        """
        Get capacity of parking

        ### Returns
            capacity (`int`): The total parking spaces
        """
        return self._capacity

    def get_occupied_spaces(self):
        """
        Get a list of the filled parking spaces

        ### Returns
            ParkingSpace[] : The occupied parking spaces
        """
        return list(filter(lambda ps: not ps.is_free(), self._parking_spaces))

    def get_free_spaces(self):
        """
        Get a list of the available parking spaces

        ### Returns
            ParkingSpace[] : The free parking spaces
        """
        return list(filter(lambda ps: ps.is_free(), self._parking_spaces))

    def assign_vehicle(self, vehicle: Vehicle):
        """
        Assign vehicle to a random parking space

        ### Arguments:
            vehicle (``Vehicle``) :
                description: A non-parked instance of the vehicle class

        ### Raises:
            ParkingIsFull: The parking has no free spaces
        """
        free_spaces = self.get_free_spaces()
        if len(free_spaces) == 0:
            raise ParkingIsFull()

        random.choice(free_spaces).assign_vehicle(vehicle)

    def toJson(self) -> Dict[str, Any]:
        return {
            "class": Parking.__name__,
            "free_parking_spaces": list(map(ParkingSpace.toJson, self.get_free_spaces())),
            "occupied_parking_spaces": list(map(ParkingSpace.toJson, self.get_occupied_spaces())),
        }

    def __repr__(self) -> str:
        return json.dumps(self.toJson(), indent=4)
