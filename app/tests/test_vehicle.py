from app.utils import find_intersection, find_intersection_v2
import numpy as np
from config import MAX_CHARGING_RATE, MAX_DISCHARGING_RATE
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import timeit
from app.models.vehicle import Vehicle


def test_vehicle(vehicle: Vehicle, axes: Axes):
    max_c, min_c = vehicle._calculate_charge_curves()
    # data = [(i, max_c[i], min_c[i]) for i in range(vehicle.get_time_before_departure() + 1)]

    # print(vehicle)
    # print(data)

    axes.plot(max_c)
    axes.plot(min_c)


def benchmark_update():
    v = Vehicle(10, 55, 10, 60, 0, "bench")
    v.park(MAX_CHARGING_RATE, MAX_DISCHARGING_RATE)

    x_axes = range(0, v._time_before_departure + 1)

    t1 = timeit.timeit(lambda: v._calculate_charge_curves(), number=1000)
    max_curve, min_curve = v._calculate_charge_curves()
    current_charge_line = [v._current_charge for _ in x_axes]

    t2 = timeit.timeit(lambda: np.trapz(max_curve), number=1000)

    t3 = timeit.timeit(
        lambda: find_intersection(x_axes[1:], max_curve[1:], x_axes[1:], current_charge_line[1:]), number=1000
    )

    t4 = timeit.timeit(
        lambda: find_intersection_v2(x_axes, max_curve, v._current_charge), number=1000
    )

    intersection = find_intersection_v2(x_axes, max_curve, v._current_charge)
    print(intersection)
    intersection = find_intersection_v2(x_axes, min_curve, v._current_charge)
    print(intersection)

    intersection = find_intersection(x_axes[1:], max_curve[1:], x_axes[1:], current_charge_line[1:])
    print(intersection)
    intersection = find_intersection(x_axes[1:], min_curve[1:], x_axes[1:], current_charge_line[1:])
    print(intersection)

    print(f"Calculate charge curve: {t1} msec")
    print(f"Calculate area using np.trapz: {t2} msec")
    print(f"Find intersection using third party: {t3} msec")
    print(f"Find intersection using custom: {t4} msec")

    plt.figure()
    test_vehicle(v, plt)
    plt.show()
