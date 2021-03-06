from app.models.environment import V2GEnvironment
import matplotlib.pyplot as plt

from app.tests.test_vehicle import test_vehicle

env = V2GEnvironment(5, 'data/GR-data-new.csv', 'test')


def create_vehicle_diagram(step: int):
    fig, axes = plt.subplots(1, 5, sharey="row")
    fig.set_size_inches(18, 9)
    axes = axes.flatten()

    for i, v in enumerate(env._parking._vehicles):
        test_vehicle(v, axes[i])

    fig.savefig(f"plots/step_{step}.png")
    plt.close(fig)


obs = env._reset()

print(f'Observation: {obs}')
print(env._parking)
create_vehicle_diagram(env._state["step"])
while not obs.is_last():
    action = int(input("Provide coefficient: "))
    obs = env.step(action)
    print(f'Observation: {obs}')
    print(env._parking)
    create_vehicle_diagram(env._state["step"])
