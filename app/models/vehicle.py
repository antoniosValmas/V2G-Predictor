from app.utils import find_intersection
import json
import numpy as np
import math
from typing import Any, Dict


class Vehicle:
    """
    A class to represent a vehicle and its charging functionalities

    ### Arguments:
        initial_charge (``float``) :
            description: The initial charge of the vehicle's battery
        target_charge (``float``) :
            description: The desired charge to be achieved in `total_stay` hours
        total_stay (``int``) :
            description: The total hours that the vehicle will remain in the parking
        max_charge (``float``) :
            description: The maximum allowed charge to be stored in the vehicle's battery
        min_charge (``float``) :
            description: The minimum allowed charge to be stored in the vehicle's battery

    ### Attributes:
        _charge_priority (``float``) :
            description: The charge priority of the vehicle
        _discharge_priority (``float``) :
            description: The discharge priority of the vehicle
        _next_max_charge (``float``) :
            description: The maximum charging state needed to be achieved in the next hour so tha
            the target_charge is achievable
        _next_min_charge (``float``) :
            description: The minimum charging state needed to be achieved in the next hour so that
            the target_charge is achievable
    """

    def __init__(
        self,
        initial_charge: float,
        target_charge: float,
        total_stay: int,
        max_charge: float,
        min_charge: float,
    ):
        self._current_charge = initial_charge
        self._target_charge = target_charge
        self._time_before_departure = total_stay
        self._max_charge = max_charge
        self._min_charge = min_charge
        self._charge_priority = 0.0
        self._discharge_priority = 0.0

    def park(self, max_charging_rate: float, max_discharging_rate: float):
        """
        Park vehicle to parking space and initialize its state variables
        To be called by the parking space it was assigned to

        ### Arguments:
            max_charging_rate (``float``):
                description: The maximum charging rate
            max_discharging_rate (``float``):
                description: The maximum discharging rate
        """
        self._max_charging_rate = max_charging_rate
        self._max_discharging_rate = max_discharging_rate

        self.update()

    def _calculate_next_max_charge(self, current_charge: float, time_before_departure: int):
        """
        Calculate next max charge based on the below formula

        min (
            ``max_charging_rate (kWh) * 1 hour + current_charge,``\n
            ``max_charge,``\n
            ``(time_of_departure - time_has_past - 1) * max_discharging_rate + target_charge,``\n
        )

        Note: ``(time_of_departure - time_has_past - 1) = time_before_departure - 1``

        ### Arguments:
            current_charge (``float``):
                description: The current charge of the vehicle's battery
            time_before_departure (``int``):
                description: The total hours remaining before departure

        ### Returns:
            float : The next max charge
        """
        return min(
            self._max_charging_rate + current_charge,
            self._max_charge,
            (time_before_departure - 1) * self._max_discharging_rate + self._target_charge,
        )

    def _calculate_next_min_charge(self, current_charge: float, time_before_departure: int):
        """
        Calculate next min charge based on the below formula

        max (
            ``current_charge - max_discharging_rate (kWh) * 1 hour,``\n
            ``min_charge,``\n
            ``target_charge - (time_of_departure - time_has_past - 1) * max_charging_rate,``\n
        )

        Note: ``(time_of_departure - time_has_past - 1) = time_before_departure - 1``

        ### Arguments:
            current_charge (``float``):
                description: The current charge of the vehicle's battery
            time_before_departure (``int``):
                description: The total hours remaining before departure

        ### Returns:
            float : The next min charge
        """
        return max(
            current_charge - self._max_discharging_rate,
            self._min_charge,
            self._target_charge - (time_before_departure - 1) * self._max_charging_rate,
        )

    def _update_next_charging_states(self):
        """
        Update next max and min charge state variables

        Also calculate the emergency charges:
            - If both next charges are positive then:
                ``emergency_plus_charge = next_min_charge - current_charge, emergency_minus_charge = 0``
            - if both next charges are negative then:
                ``emergency_minus_charge = next_max_charge - current_charge, emergency_plus_charge = 0``
            - Otherwise ``emergency_minus_charge = emergency_plus_charge = 0``
        """
        self._next_max_charge = self._calculate_next_max_charge(self._current_charge, self._time_before_departure)
        self._next_min_charge = self._calculate_next_min_charge(self._current_charge, self._time_before_departure)
        if self._next_max_charge > 0 and self._next_min_charge > 0:
            self._emergency_plus_charge = self._next_min_charge - self._current_charge
            self._emergency_minus_charge = 0.
        elif self._next_max_charge < 0 and self._next_min_charge < 0:
            self._emergency_plus_charge = 0.
            self._emergency_minus_charge = self._next_max_charge - self._current_charge
        else:
            self._emergency_plus_charge = 0.
            self._emergency_minus_charge = 0.

    def _calculate_charge_curves(self):
        """
        Calculate the max and min charge curves of the vehicle

        The max curve is the curve describing the maximum possible charge the vehicle can achieve at each timestamp
        so that the target charge is achievable given a initial charge, a max charging rate, a max discharging rate,
        a max charge and the time before departure

        The min curve is the curve describing the minimum possible charge the vehicle can achieve at each timestamp
        so that the target charge is achievable given a initial charge, a max charging rate, a max discharging rate,
        a min charge and the time before departure

        ### Returns
            Tuple[float[], float[]] : The points of the max and min curve respectively in ascending time order
        """
        current_max_charge = self._current_charge
        current_min_charge = self._current_charge
        max_charges = [current_max_charge]
        min_charges = [current_min_charge]
        for t in range(self._time_before_departure, 0, -1):
            current_max_charge = self._calculate_next_max_charge(current_max_charge, t)
            current_min_charge = self._calculate_next_min_charge(current_min_charge, t)

            max_charges.append(current_max_charge)
            min_charges.append(current_min_charge)

        return max_charges, min_charges

    def _update_priorities(self):
        """
        Update the charging and discharging priorities of the vehicle

        The vehicle priorities express the vehicle's need of charging or discharging

        The priorities are calculated as follows:
        - First we find the area included between the max and min curves (max/min area)
        - The charging priority is calculated as the ratio of
            - the area above the y = current_charge line that is part of the "max/min area"
            - and the "max/min area"
        - The discharging priority is calculated as the ratio of
            - the area below the y = current_charge line that is part of the "max/min area"
            - and the "max/min area"

        From the above it is obvious that the following is true for the two priorities:
        ``charging_priority = 1 - discharging_priority``
        """
        x_axes = range(0, self._time_before_departure + 1)
        max_curve, min_curve = self._calculate_charge_curves()
        current_charge_line = [self._current_charge for _ in x_axes]

        max_curve_area: float = np.trapz(max_curve)
        min_curve_area: float = np.trapz(min_curve)
        diff_curve_area = max_curve_area - min_curve_area

        max_intersection = find_intersection(x_axes, max_curve, x_axes, current_charge_line)
        min_intersection = find_intersection(x_axes, min_curve, x_axes, current_charge_line)

        intersection = max_intersection or min_intersection
        if (intersection is None) and (self._current_charge == self._target_charge):
            intersection = (float(self._time_before_departure), self._current_charge)

        if intersection is None:
            diff = self._current_charge - self._target_charge
            self._charge_priority = 0 if diff > 0 else 1
            self._discharge_priority = 1 if diff > 0 else 0
        else:
            inter_x, inter_y = intersection
            cutoff_index = math.ceil(inter_x)
            partial_min_curve = min_curve[:cutoff_index]
            partial_min_curve.append(inter_y)
            partial_x_axes = list(range(0, cutoff_index))
            partial_x_axes.append(inter_x)

            current_charge_area: float = np.trapz([self._current_charge, inter_y], [0, inter_x])
            area_bellow: float = np.trapz(partial_min_curve, partial_x_axes)

            discharge_area = current_charge_area - area_bellow
            self._discharge_priority = discharge_area / diff_curve_area
            self._charge_priority = 1 - self._discharge_priority

    def update(self):
        """
        Update state variables
        """
        self._update_next_charging_states()
        self._update_priorities()

    def update_current_charge(self, energy: float, mean_priority: float, residue_energy: float):
        """
        Update current charge by providing:
            - the total energy, gained or lost, divided by the number of cars in the parking
            - the mean charge/discharge priority
            - any residue energy that wasn't allocated by the previous vehicles

        The current charge is updated based on the following formula:
            ``current_charge = current_charge + energy *
            (1 + charge/discharge_priority - mean_priority) + residue_energy``

        ### Arguments:
            energy (``float``) :
                description: The total energy, bought or sold in this timestep, divided by the number
                of cars in the parking
            mean_priority (``float``) :
                description: The mean charge or discharge priority (if we bought or sold energy respectively)
            residue_energy (``float``) :
                description: Any residue energy that wasn't allocated by the previous vehicles

        ### Returns:
            float : The residue energy that wasn't allocated by this vehicle
        """
        charging = energy > 0
        priority = self._charge_priority if charging else self._discharge_priority

        self._current_charge = self._current_charge + energy * (1 + priority - mean_priority) + residue_energy

        residue = max(self._current_charge - self._max_charge, 0) + min(self._current_charge - self._min_charge, 0)
        self._current_charge = min(self._current_charge, self._max_charge)
        self._current_charge = max(self._current_charge, self._min_charge)

        self._time_before_departure -= 1

        if self._time_before_departure != 0:
            self.update()

        return residue

    def update_emergency_demand(self, energy):
        if energy < 0:
            self._target_charge -= (self._emergency_plus_charge - energy)
        else:
            self._target_charge -= (self._emergency_minus_charge - energy)

    def get_current_charge(self):
        """
        Get current charge

        ### Returns:
            float : The current battery's charge
        """
        return self._current_charge

    def get_target_charge(self):
        """
        Get the target charge

        ### Returns:
            float : The desired charge to be achieved before departure
        """
        return self._target_charge

    def get_time_before_departure(self):
        """
        Get the total time before departure

        ### Returns:
            int : The total time remaining before departure
        """
        return self._time_before_departure

    def get_max_charge(self):
        """
        Get max charge

        ### Returns:
            float : The maximum allowed charge for the vehicle's battery
        """
        return self._max_charge

    def get_min_charge(self):
        """
        Get min charge

        ### Returns:
            float : The minimum allowed charge for the vehicle's battery
        """
        return self._min_charge

    def get_next_max_charge(self):
        """
        Get next max charge

        ### Returns:
            float : The next maximum charge that can be achieved without compromising the target
        """
        return self._next_max_charge

    def get_next_min_charge(self):
        """
        Get next min charge

        ### Returns:
            float : The next minimum charge that can be achieved without compromising the target
        """
        return self._next_min_charge

    def get_charge_priority(self):
        """
        Get charge priority

        ### Returns:
            float : The charge priority of the vehicle
        """
        return self._charge_priority

    def get_discharge_priority(self):
        """
        Get discharge priority

        ### Returns:
            float : The discharge priority of the vehicle
        """
        return self._discharge_priority

    def get_emergency_plus_charge(self):
        """
        Get emergency plus charge of the vehicle

        ### Returns:
            float : The emergency plus charge of the vehicle
        """
        return self._emergency_plus_charge

    def get_emergency_minus_charge(self):
        """
        Get emergency minus charge of the vehicle

        ### Returns:
            float : The emergency minus charge of the vehicle
        """
        return self._emergency_minus_charge

    def toJson(self) -> Dict[str, Any]:
        return {
            "class": Vehicle.__name__,
            "current_change": self._current_charge,
            "target_charge": self._target_charge,
            "time_before_departure": self._time_before_departure,
            "max_charge": self._max_charge,
            "min_charge": self._min_charge,
            "max_charging_rate": self._max_charging_rate,
            "min_discharging_rate": self._max_discharging_rate,
            "next_max_charge": self._next_max_charge,
            "next_min_charge": self._next_min_charge,
            "charge_priority": self._charge_priority,
            "discharge_priority": self._discharge_priority,
        }

    def __repr__(self) -> str:
        return json.dumps(self.toJson(), indent=4)