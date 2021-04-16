from app.models.environment import V2GEnvironment
import matplotlib.pyplot as plt

from app.tests.test_vehicle import test_vehicle

env = V2GEnvironment(10, 'data/GR-data-11-20.csv')


def create_vehicle_diagram(step: int):
    fig, axes = plt.subplots(2, 5, sharey="row")
    fig.set_size_inches(18, 9)
    axes = axes.flatten()

    for i, v in enumerate(env._parking._vehicles):
        test_vehicle(v, axes[i])

    fig.savefig(f"plots/step_{step}.png")
    plt.close(fig)


obs = env._reset()

print(f'Observation: {obs}')
print(env._parking)
create_vehicle_diagram(env._state["elapsed_time"])
while not obs.is_last():
    a = int(input("Provide coefficient: "))
    action = [i == a for i in range(env._length)]
    obs = env._step(action)
    print(f'Observation: {obs}')
    print(env._parking)
    create_vehicle_diagram(env._state["elapsed_time"])
