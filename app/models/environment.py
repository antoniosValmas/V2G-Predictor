from app.error_handling import ParkingIsFull
from typing import List
import numpy as np
import random
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step

from app.models.parking import Parking
from app.models.vehicle import Vehicle


class V2GEnvironment(PyEnvironment):
    _precision = 3
    _length = _precision * 2 + 1
    _charging_coefficient_map = np.linspace(-1.0, 1.0, num=_length)

    def __init__(self, capacity: int):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(self._length,), dtype=np.int, minimum=0, maximum=1, name="action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(9,),
            dtype=np.float,
            minimum=(0, -np.inf, -np.inf, 0, 0, 0, 0, 0, 0),
            maximum=(np.inf, np.inf, np.inf, 1, 1, 1, 1, 1, 1),
        )
        self._state = {"elapsed_time": 0}
        self._parking = Parking(capacity)
        self._add_new_cars(offset=2)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self) -> time_step.TimeStep:
        self._state = {"elapsed_time": 0}
        return time_step.TimeStep(
            time_step.StepType.FIRST,
            0,
            0,
            [
                0,
                0,
                0,
                self._parking.get_next_max_charge(),
                self._parking.get_next_min_charge(),
                self._parking.get_next_max_discharge(),
                self._parking.get_next_min_discharge(),
                self._parking.get_charge_mean_priority(),
                self._parking.get_discharge_mean_priority()
            ],
        )

    def _step(self, action: List[int]) -> time_step.TimeStep:
        idx = action.index(1)
        charging_coefficient = self._charging_coefficient_map[idx]
        print("a = ", charging_coefficient)
        self._parking.update(charging_coefficient)
        self._add_new_cars()
        self._state["elapsed_time"] += 1

        return time_step.TimeStep(
            time_step.StepType.MID if self._state["elapsed_time"] < 24 else time_step.StepType.LAST,
            0,
            1,
            [
                0,
                0,
                0,
                self._parking.get_next_max_charge(),
                self._parking.get_next_min_charge(),
                self._parking.get_next_max_discharge(),
                self._parking.get_next_min_discharge(),
                self._parking.get_charge_mean_priority(),
                self._parking.get_discharge_mean_priority()
            ],
        )

    def _add_new_cars(self, offset=0):
        new_cars = random.choice([0, 0, 0, 0, 0, 1, 1, 2])
        try:
            for _ in range(new_cars + offset):
                v = self._create_vehicle()
                self._parking.assign_vehicle(v)
        except ParkingIsFull:
            print("Parking is full no more cars added")

    def _create_vehicle(self):
        total_stay = random.randint(7, 10)
        min_charge = 10
        max_charge = 90
        initial_charge = round(10 + random.random() * 50, 3)
        target_charge = round(40 + random.random() * 50, 3)

        return Vehicle(initial_charge, target_charge, total_stay, max_charge, min_charge)
