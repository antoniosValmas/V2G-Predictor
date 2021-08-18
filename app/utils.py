import math
import random
from typing import List, Tuple, Union
from intersect import intersection
import numpy as np


def find_intersection(x1, y1, x2, y2) -> Union[Tuple[float, float], None]:
    x, y = intersection(x1, y1, x2, y2)
    intersections = list(filter(lambda val: val[0] > 0, set(zip(x, y))))

    if len(intersections) == 0:
        return None

    return intersections[0]


def find_intersection_v2(x1, y1, c) -> Union[Tuple[float, float], None]:
    length = len(y1)
    for i in range(length - 2, -1, -1):
        if y1[i + 1] == c:
            return x1[i + 1], c
        elif y1[i] > c > y1[i + 1] or y1[i] < c < y1[i + 1]:
            y = abs(y1[i] - y1[i + 1])
            y_ = abs(c - y1[i + 1])
            x = abs(x1[i + 1] - x1[i])
            x_ = (1 - y_ / y) * x
            return x1[i] + x_, c

    return None


def create_vehicle_distribution(steps=24 * 30 * 6) -> List[List[int]]:
    time_of_day = 0
    vehicles = []
    for _ in range(steps):
        day_coefficient = math.sin(math.pi / 6 * time_of_day) / 2 + 0.5
        new_cars = max(0, int(np.random.normal(20 * day_coefficient, 2 * day_coefficient)))
        time_of_day += 1
        time_of_day %= 24
        if time_of_day > 21:
            vehicles.append([])
        else:
            vehicles.append(
                [
                    (
                        min(
                            24 - time_of_day,
                            random.randint(7, 12),
                        ),
                        round(6 + random.random() * 20, 2),
                        round(34 + random.random() * 20, 2),
                    )
                    for _ in range(new_cars)
                ]
            )

    return vehicles
