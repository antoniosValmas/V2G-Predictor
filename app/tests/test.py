import random
import matplotlib.pyplot as plt

from app.tests.test_vehicle import test_vehicle
from app.models.vehicle import Vehicle
from app.models.parking import Parking

parking = Parking(10)


def create_vehicle():
    total_stay = random.randint(7, 10)
    min_charge = 10
    max_charge = 90
    initial_charge = 10 + random.random() * 50
    target_charge = 40 + random.random() * 50

    return Vehicle(initial_charge, target_charge, total_stay, max_charge, min_charge)


vehicles = []
fig, axes = plt.subplots(2, 5, sharey='row')
fig.set_size_inches(18, 9)
axes = axes.flatten()

for i in range(10):
    v = create_vehicle()
    vehicles.append(v)
    parking.assign_vehicle(v)
    test_vehicle(v, axes[i])
    print(v)

plt.savefig('before.png')
fig, axes = plt.subplots(2, 5, sharey='row')
fig.set_size_inches(18, 9)
axes.flatten()
axes = axes.flatten()

# print(parking)
parking.update_energy_state(0.5)
print('After')

for i, v in enumerate(vehicles):
    test_vehicle(v, axes[i])
    print(v)

plt.savefig('after.png')
