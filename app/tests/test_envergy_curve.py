from app.models.energy import EnergyCurve
import matplotlib.pyplot as plt

data = [
    (0, 10.0),
    (1, 9.0),
    (2, 7.0),
    (3, 6.0),
    (4, 5.5),
    (5, 5.0),
    (6, 5.0),
    (7, 6.0),
    (8, 8.0),
    (9, 9.5),
]

ec = EnergyCurve(data)

x = ec.get_x()
y = ec.get_y()
dy_dx = ec.get_first_derivate()
dy_dx2 = ec.get_second_derivate()

plt.plot(x, y, label="y")
plt.plot(x, dy_dx, label="dy/dx")
plt.plot(x, dy_dx2, label="$d^2y/dx^2$")
plt.legend()
_ = plt.xlabel("x")
plt.show()
