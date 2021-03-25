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
        """
        Returns True if no vehicles are assigned to the parking space, false otherwise

        ### Returns
            boolean : Whether it is free or not
        """
        return self._vehicle is None

    def get_vehicle(self):
        """
        Get the assigned vehicle. Returns None if no vehicle are assigned

        ### Returns:
            Vehicle: The assigned vehicle. None if no vehicle are assigned
        """
        return self._vehicle

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

    def update_energy_state(self, charging_coefficient: float, mean_priority: float, residue: float):
        """
        Update the energy state of the parked vehicle

        ### Arguments:
            charging_coefficient (``float``) :
                description: The ratio of the used charging/discharging capacity
            mean_priority (``float``) :
                description: The mean priority of all vehicles
            residue (``float``) :
                description: The residue energy from the previous vehicle
        """
        a = charging_coefficient
        energy = self._max_charging_rate * a if a > 0 else self._max_discharging_rate * a
        return self._vehicle.update_current_charge(energy, mean_priority, residue)

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

    _occupied_parking_spaces: Dict[int, ParkingSpace] = {}
    _next_max_charge: float = 0.0
    _next_min_charge: float = 0.0
    _current_charge: float = 0.0

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._free_parking_spaces = {(i + 1): ParkingSpace() for i in range(capacity)}

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
        return self._occupied_parking_spaces

    def get_free_spaces(self):
        """
        Get a list of the available parking spaces

        ### Returns
            ParkingSpace[] : The free parking spaces
        """
        return self._free_parking_spaces

    def get_current_energy(self):
        return self._current_charge

    def get_next_max_charge(self):
        """
        Get next cumulative maximum charge

        ### Returns:
            float : The sum of all vehicles next max charge
        """
        return self._next_max_charge

    def get_next_min_charge(self):
        """
        Get next cumulative minimum charge

        ### Returns:
            float : The sum of all vehicles next min charge
        """
        return self._next_min_charge

    def get_emergency_plus_charge(self):
        """
        Get cumulative emergency plus charge

        ### Returns:
            float : The cumulative emergency plus charge
        """
        return sum(
            [space.get_vehicle().get_emergency_plus_charge() for space in self._occupied_parking_spaces.values()]
        )

    def get_emergency_minus_charge(self):
        """
        Get cumulative emergency minus charge

        ### Returns:
            float : The cumulative emergency minus charge
        """
        return sum(
            [space.get_vehicle().get_emergency_minus_charge() for space in self._occupied_parking_spaces.values()]
        )

    def assign_vehicle(self, vehicle: Vehicle):
        """
        Assign vehicle to a random parking space

        ### Arguments:
            vehicle (``Vehicle``) :
                description: A non-parked instance of the vehicle class

        ### Raises:
            ParkingIsFull: The parking has no free spaces
        """
        self._current_charge += vehicle.get_current_charge()
        free_spaces = self._free_parking_spaces
        num_free_spaces = len(free_spaces.values())
        if num_free_spaces == 0:
            raise ParkingIsFull()

        index = random.choice(list(free_spaces.keys()))
        self._occupied_parking_spaces[index] = free_spaces[index]
        free_spaces.pop(index)
        self._occupied_parking_spaces[index].assign_vehicle(vehicle)

    def depart_vehicles(self):
        occupied_spaces = self._occupied_parking_spaces.items()
        for idx, space in occupied_spaces:
            if space.get_vehicle().get_time_before_departure() == 0:
                self._current_charge -= space.get_vehicle().get_current_charge()
                space.remove_vehicle()
                self._free_parking_spaces[idx] = space
                self._occupied_parking_spaces.pop(idx)

    def update_next_charges(self):
        """
        Update next cumulative max and min charges
        """
        occupied_spaces = self._occupied_parking_spaces.values()
        self._next_max_charge = 0
        self._next_min_charge = 0
        for space in occupied_spaces:
            self._next_max_charge += space.get_vehicle().get_next_max_charge()
            self._next_min_charge += space.get_vehicle().get_next_min_charge()

    def update_energy_state(self, charging_coefficient):
        """
        Update energy state of parking

        ### Arguments:
            charging_coefficient (``float``) :
                description: The ratio of the used charging/discharging capacity
        """
        next_energy_state = (
            self._next_max_charge - self._current_charge
            if charging_coefficient > 0
            else self._current_charge - self._next_min_charge
        )

        available_energy = next_energy_state * charging_coefficient

        emergency_charge = (
            Vehicle.get_emergency_plus_charge if charging_coefficient > 0 else Vehicle.get_emergency_plus_charge
        )
        priority = Vehicle.get_charge_priority if charging_coefficient > 0 else Vehicle.get_discharge_priority

        occupied_spaces = sorted(self._occupied_parking_spaces.values(), key=lambda s: priority(s), reverse=True)
        total_priority = 0.0

        for space in occupied_spaces:
            v = space.get_vehicle()
            total_priority += priority(v)

            v.update_emergency_demand(min(available_energy, emergency_charge(v)))
            available_energy = max(0, available_energy - emergency_charge(v))

        charging_coefficient = available_energy / next_energy_state
        mean_priority = total_priority / len(occupied_spaces)
        residue = 0.0
        for space in occupied_spaces:
            residue = space.update_energy_state(charging_coefficient, mean_priority, residue)

    def toJson(self) -> Dict[str, Any]:
        return {
            "class": Parking.__name__,
            "free_parking_spaces": list(map(ParkingSpace.toJson, self.get_free_spaces())),
            "occupied_parking_spaces": list(map(ParkingSpace.toJson, self.get_occupied_spaces())),
        }

    def __repr__(self) -> str:
        return json.dumps(self.toJson(), indent=4)
