import json
from typing import Any, Dict, List
import config as cfg
from app.error_handling import ParkingIsFull
from app.models.vehicle import Vehicle


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

    _next_max_charge: float = 0.0
    _next_min_charge: float = 0.0
    _next_max_discharge: float = 0.0
    _next_min_discharge: float = 0.0
    _charge_mean_priority: float = 0.0
    _discharge_mean_priority: float = 0.0
    _current_charge: float = 0.0
    _max_charging_rate = cfg.MAX_CHARGING_RATE
    _max_discharging_rate = cfg.MAX_DISCHARGING_RATE
    _vehicles: List[Vehicle] = []

    def __init__(self, capacity: int):
        self._capacity = capacity

    def get_capacity(self):
        """
        Get capacity of parking

        ### Returns
            capacity (`int`): The total parking spaces
        """
        return self._capacity

    def get_current_energy(self):
        """
        Get total energy stored in the parking

        ### Returns:
            float : The sum of all vehicles' current energy
        """
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

    def get_next_max_discharge(self):
        """
        Get next cumulative maximum discharge

        ### Returns:
            float : The sum of all vehicles next max discharge
        """
        return self._next_max_discharge

    def get_next_min_discharge(self):
        """
        Get next cumulative minimum discharge

        ### Returns:
            float : The sum of all vehicles next min discharge
        """
        return self._next_min_discharge

    def get_charge_mean_priority(self):
        """
        Get mean charge priority
        ### Returns
            float : The mean charge priority
        """
        return self._charge_mean_priority

    def get_discharge_mean_priority(self):
        """
        Get mean discharge priority
        ### Returns
            float : The mean discharge priority
        """
        return self._discharge_mean_priority

    def get_max_charging_rate(self):
        """
        Get max charging rate
        ### Returns
            float : The max charging rate
        """
        return self._max_charging_rate

    def get_max_discharging_rate(self):
        """
        Get max discharging rate
        ### Returns
            float : The max discharging rate
        """
        return self._max_discharging_rate

    def assign_vehicle(self, vehicle: Vehicle):
        """
        Assign vehicle to a parking space

        ### Arguments:
            vehicle (``Vehicle``) :
                description: A non-parked instance of the vehicle class

        ### Raises:
            ParkingIsFull: The parking has no free spaces
        """
        self._current_charge += vehicle.get_current_charge()
        num_of_vehicles = len(self._vehicles)
        if num_of_vehicles == self._capacity:
            raise ParkingIsFull()

        vehicle.park(self._max_charging_rate, self._max_discharging_rate)
        self._vehicles.append(vehicle)
        self._vehicles.sort(key=lambda v: v.get_time_before_departure())
        self._update_parking_state()

    def depart_vehicles(self):
        """
        Filter out all vehicles that have left the parking
        """
        self._vehicles = list(filter(lambda v: v.get_time_before_departure() != 0, self._vehicles))

    def _update_parking_state(self):
        self._next_max_charge = 0.0
        self._next_min_charge = 0.0
        self._next_max_discharge = 0.0
        self._next_min_discharge = 0.0
        self._charge_mean_priority = 0.0
        self._discharge_mean_priority = 0.0
        self._normalization_charge_constant = 0.0
        self._normalization_discharge_constant = 0.0
        for vehicle in self._vehicles:
            next_max_charge = vehicle.get_next_max_charge()
            next_min_charge = vehicle.get_next_min_charge()
            next_max_discharge = vehicle.get_next_max_discharge()
            next_min_discharge = vehicle.get_next_min_discharge()
            self._next_max_charge += next_max_charge
            self._next_min_charge += next_min_charge
            self._next_max_discharge += next_max_discharge
            self._next_min_discharge += next_min_discharge

            charge_priority = vehicle.get_charge_priority()
            discharge_priority = vehicle.get_discharge_priority()
            self._charge_mean_priority += charge_priority
            self._discharge_mean_priority += discharge_priority

            self._normalization_charge_constant += next_max_charge * charge_priority
            self._normalization_discharge_constant += next_max_discharge * discharge_priority

        number_of_vehicles = len(self._vehicles)

        if number_of_vehicles == 0:
            return

        self._charge_mean_priority = round(self._charge_mean_priority / number_of_vehicles, 3)
        self._discharge_mean_priority = round(self._discharge_mean_priority / number_of_vehicles, 3)

        if self._next_max_charge != 0:
            self._normalization_charge_constant = round(self._normalization_charge_constant / self._next_max_charge, 3)

        if self._next_max_discharge != 0:
            self._normalization_discharge_constant = round(
                self._normalization_discharge_constant / self._next_max_discharge, 3
            )

    def update_energy_state(self, charging_coefficient: float):
        """
        Update energy state of parking

        ### Arguments:
            charging_coefficient (``float``) :
                description: The ratio of the used charging/discharging capacity
        """
        is_charging = charging_coefficient > 0
        sign = 1 if is_charging else -1
        next_energy_diff = self._next_max_charge if is_charging else self._next_max_discharge

        available_energy = next_energy_diff * abs(charging_coefficient)

        min_energy = Vehicle.get_next_min_charge if is_charging else Vehicle.get_next_min_discharge
        priority = Vehicle.get_charge_priority if is_charging else Vehicle.get_discharge_priority
        normalization_constant = (
            self._normalization_charge_constant if is_charging else self._normalization_discharge_constant
        )

        priority_sorted_vehicles = sorted(self._vehicles, key=lambda s: priority(s), reverse=True)

        for vehicle in priority_sorted_vehicles:
            vehicle.update_emergency_demand(min(available_energy, min_energy(vehicle)), is_charging)
            available_energy = max(0, available_energy - min_energy(vehicle))

        charging_coefficient = round(available_energy / next_energy_diff, 3) if next_energy_diff != 0 else 0
        residue = 0.0
        avg_charge_levels: List[float] = []
        for vehicle in sorted(
            priority_sorted_vehicles,
            key=lambda v: charging_coefficient * (1 + priority(v) - normalization_constant),
            reverse=True,
        ):
            avg_charge_level, residue = vehicle.update_current_charge(
                sign * charging_coefficient, normalization_constant, residue
            )
            avg_charge_levels.append(avg_charge_level)

        return avg_charge_levels

    def update(self, charging_coefficient):
        """
        Given the action input, it performs an update step

        ### Arguments:
            charging_coefficient (``float``) :
                description: The charging coefficient
        """
        avg_charge_levels = self.update_energy_state(charging_coefficient)
        self.depart_vehicles()
        self._update_parking_state()
        return avg_charge_levels

    def toJson(self) -> Dict[str, Any]:
        return {
            "class": Parking.__name__,
            "max_charging_rage": self._max_charging_rate,
            "max_discharging_rate": self._max_discharging_rate,
            "vehicles": list(map(lambda v: v.toJson(), self._vehicles)),
        }

    def __repr__(self) -> str:
        return json.dumps(self.toJson(), indent=4)
