from matplotlib.axes import Axes

from app.models.vehicle import Vehicle


def test_vehicle(vehicle: Vehicle, axes: Axes):
    max_c, min_c = vehicle._calculate_charge_curves()
    # data = [(i, max_c[i], min_c[i]) for i in range(vehicle.get_time_before_departure() + 1)]

    # print(vehicle)
    # print(data)

    axes.plot(max_c)
    axes.plot(min_c)


class Testing():
    test_array = []

    def __init__(self, t):
        self.test_array.append(t)
