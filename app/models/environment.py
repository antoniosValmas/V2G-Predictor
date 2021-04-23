import json
from typing import Any, Dict
from app.models.energy import EnergyCurve
from app.error_handling import ParkingIsFull
import numpy as np
import random
import math
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step

from app.models.parking import Parking
from app.models.vehicle import Vehicle


class V2GEnvironment(PyEnvironment):
    _precision = 10
    _length = _precision * 2 + 1
    _charging_coefficient_map = np.linspace(-1.0, 1.0, num=_length)
    _battery_cost = 120
    _battery_capacity = 60

    def __init__(self, capacity: int, dataFile: str, name: str):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self._length - 1, name="action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(30,), dtype=np.float, minimum=0, maximum=1, name="observation"
        )
        self._time_step_spec = time_step.TimeStep(
            step_type=array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=2),
            discount=array_spec.BoundedArraySpec(shape=(), dtype=np.float, minimum=0.0, maximum=1.0),
            reward=array_spec.ArraySpec(shape=(), dtype=np.float),
            observation=self._observation_spec,
        )
        self.name = name
        self._state = {"time_of_day": 0, "step": 0}
        self._parking = Parking(capacity, name)
        self._energy_curve = EnergyCurve(dataFile)
        self._add_new_cars()
        self._reset()

    def time_step_spec(self):
        return self._time_step_spec

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self) -> time_step.TimeStep:
        self._state = {"time_of_day": 0, "step": 0}

        self._energy_curve.reset()
        energy_costs, _ = self._energy_curve.get_next_batch()
        self._current_time_step = time_step.TimeStep(
            step_type=time_step.StepType.FIRST,
            reward=0.0,
            discount=1.0,
            observation=np.array(
                [
                    *energy_costs,
                    self._parking.get_next_max_charge() / self._parking.get_max_charging_rate(),
                    self._parking.get_next_min_charge() / self._parking.get_max_charging_rate(),
                    self._parking.get_next_max_discharge() / self._parking.get_max_discharging_rate(),
                    self._parking.get_next_min_discharge() / self._parking.get_max_discharging_rate(),
                    self._parking.get_charge_mean_priority(),
                    self._parking.get_discharge_mean_priority(),
                ],
                dtype=np.float,
            ),
        )
        return self._current_time_step

    def _step(self, action: int) -> time_step.TimeStep:
        idx = int(action)
        charging_coefficient = self._charging_coefficient_map[idx]
        is_charging = charging_coefficient > 0

        num_of_vehicles = len(self._parking._vehicles)
        reward = 0.0
        if num_of_vehicles != 0:
            max_energy = self._parking.get_next_max_charge() if is_charging else self._parking.get_next_max_discharge()
            available_energy = max_energy * charging_coefficient
            current_cost = self._energy_curve.get_current_cost() / 1000
            min_charge = self._parking.get_next_min_charge()
            min_discharge = self._parking.get_next_min_discharge()
            max_rate = self._battery_capacity * num_of_vehicles

            avg_charge_levels = self._parking.update(charging_coefficient)

            cost = int(available_energy * current_cost * 100)
            unmet_demand = int(
                (
                    max(0, min_charge - available_energy) + min_discharge
                    if is_charging
                    else max(0, min_discharge - abs(available_energy)) + min_charge
                )
                * current_cost
                * 100
            )
            cycle_degradation_cost = int(
                self.cycle_degradation(abs(available_energy) / max_rate)
                * self._battery_capacity
                * self._battery_cost
                * num_of_vehicles
                * 100
            )
            age_degradation_cost = 0.0
            for charge_level in avg_charge_levels:
                age_degradation_cost += (
                    self.age_degradation(charge_level / self._battery_capacity)
                    * self._battery_capacity
                    * self._battery_cost
                )
            age_degradation_cost = int(age_degradation_cost * 100)

            reward = -cost - unmet_demand ** 2 - cycle_degradation_cost - age_degradation_cost

        # print(f"Energy cost: {cost}")
        # print(f"Unmet demand cost: {unmet_demand}")
        # print(f"Cycle degradation cost: {cycle_degradation_cost}")
        # print(f"Age degradation cost: {age_degradation_cost}")

        self._add_new_cars()
        self._state["time_of_day"] += 1
        self._state["time_of_day"] %= 24
        self._state["step"] += 1
        energy_costs, done = self._energy_curve.get_next_batch()

        self._current_time_step = time_step.TimeStep(
            step_type=time_step.StepType.MID if not done else time_step.StepType.LAST,
            reward=float(reward),
            discount=1.0,
            observation=np.array(
                [
                    *energy_costs,
                    self._parking.get_next_max_charge() / self._parking.get_max_charging_rate(),
                    self._parking.get_next_min_charge() / self._parking.get_max_charging_rate(),
                    self._parking.get_next_max_discharge() / self._parking.get_max_discharging_rate(),
                    self._parking.get_next_min_discharge() / self._parking.get_max_discharging_rate(),
                    self._parking.get_charge_mean_priority(),
                    self._parking.get_discharge_mean_priority(),
                ],
                dtype=np.float,
            ),
        )
        return self._current_time_step

    def _add_new_cars(self):
        day_coefficient = math.sin(math.pi / 12 * self._state["time_of_day"]) / 2 + 0.5
        new_cars = max(0, int(np.random.normal(5 * day_coefficient, 2 * day_coefficient)))
        try:
            for _ in range(new_cars):
                v = self._create_vehicle()
                self._parking.assign_vehicle(v)
        except ParkingIsFull:
            print("Parking is full no more cars added")

    def _create_vehicle(self):
        total_stay = random.randint(7, 10)
        min_charge = 6
        max_charge = 54
        initial_charge = round(10 + random.random() * 30, 3)
        target_charge = round(34 + random.random() * 20, 3)

        return Vehicle(initial_charge, target_charge, total_stay, max_charge, min_charge, self.name)

    def cycle_degradation(self, c_rate):
        return 5e-5 * c_rate + 1.8e-5

    def age_degradation(self, soc):
        """
        Returns the age degradation percentage based on the following formula

        ``L = k_t * t (in seconds) * e ^ (k_s (soc - soc_ref))``

        ### Returns
            float : the total lost capacity
        """
        return 4.14e-10 * 3600 * pow(math.e, (1.04 * (soc - 0.5)))

    def toJson(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "parking": self._parking.toJson()
        }

    def __repr__(self) -> str:
        return json.dumps(self.toJson(), indent=4)
