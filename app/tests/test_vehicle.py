import matplotlib.pyplot as plt

from app.models.vehicle import Vehicle
from app.models.parking import Parking

total_stay = 9

parking = Parking(10)
vehicle = Vehicle(15, 60, total_stay, 90, 10)

parking.assign_vehicle(vehicle)

print(parking)

max_c, min_c = vehicle._calculate_charge_curves()

data = [(i, max_c[i], min_c[i]) for i in range(total_stay + 1)]

# print(data)

# max_curve_area, min_curve_area, diff_curve_area, max_intersection, min_intersection = vehicle.update_priorities()
# print(max_curve_area)
# print(min_curve_area)
# print(diff_curve_area)
# print(max_intersection)
# print(min_intersection)

plt.plot(max_c)
plt.plot(min_c)
plt.show()
