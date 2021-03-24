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

    def park(self, max_charging_rate, max_discharging_rate):
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

        self._update_next_charging_states()
        self._update_priorities()

    def _calculate_next_max_charge(self, current_charge, time_before_departure):
        """
        Calculate next max charge based on the below formula

        min (
            ``_max_charging_rate (kWh) * 1 hour + current_charge,``\n
            ``_max_charge,``\n
            ``(time_of_departure - time_has_past - 1) * _max_discharging_rate + _target_charge,``\n
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

    def _calculate_next_min_charge(self, current_charge, time_before_departure):
        """
        Calculate next min charge based on the below formula

        max (
            ``current_charge - _max_discharging_rate (kWh) * 1 hour,``\n
            ``_min_charge,``\n
            ``_target_charge - (time_of_departure - time_has_past - 1) * _max_charging_rate,``\n
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
        """
        self._next_max_charge = self._calculate_next_max_charge(self._current_charge, self._time_before_departure)
        self._next_min_charge = self._calculate_next_min_charge(self._current_charge, self._time_before_departure)

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

        max_curve_area = np.trapz(max_curve)
        min_curve_area = np.trapz(min_curve)
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

            current_charge_area = np.trapz([self._current_charge, inter_y], [0, inter_x])
            area_bellow = np.trapz(partial_min_curve, partial_x_axes)

            discharge_area = current_charge_area - area_bellow
            self._discharge_priority = discharge_area / diff_curve_area
            self._charge_priority = 1 - self._discharge_priority

    def update_current_charge(self, energy):
        pass

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
        return self._max_charge

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

    def toJson(self) -> Dict[str, Any]:
        return {
            "class": Vehicle.__name__,
            "current_change": self._current_charge,
            "targer_charge": self._target_charge,
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
