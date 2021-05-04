from app.models.energy import EnergyCurve
import matplotlib.pyplot as plt

ec = EnergyCurve('./data/GR-data-11-20.csv', 'test')

o_x = []
o_y = []
for x, y in ec.get_raw_data():
    o_x.append(x)
    o_y.append(y)

x = ec.get_x()
y = ec.get_y()

plt.plot(x, y)
plt.plot(o_x, o_y)
plt.show()
