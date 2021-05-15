from app.policies.utils import moving_average
import matplotlib.pyplot as plt
from app.utils import create_vehicle_distribution

vehicles = create_vehicle_distribution()


plt.plot(moving_average(list(map(len, vehicles)), 24))
plt.show()

# print(vehicles)