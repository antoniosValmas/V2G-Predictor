from typing import Tuple, Union
from intersect import intersection


def find_intersection(x1, y1, x2, y2) -> Union[Tuple[float, float], None]:
    x, y = intersection(x1, y1, x2, y2)
    intersections = list(filter(lambda val: val[0] > 0, set(zip(x, y))))

    if len(intersections) == 0:
        return None

    return intersections[0]
