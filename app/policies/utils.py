from app.models.energy import EnergyCurve
import math
from typing import Dict, List, Union
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from tf_agents.environments.py_environment import PyEnvironment


def compute_avg_return(environment: PyEnvironment, policy, num_episodes=10):
    total_return = 0.0
    step = 0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            step += 1
        total_return += episode_return

    avg_return = total_return / step
    return avg_return.numpy()[0]


def sigmoid(x):
    # return 100 / (1 + math.exp(- 5 - 0.3 * x))
    print(x)
    return 1000 / (1 + math.exp(-5 - 0.0004 * x))


def plot_metric(
    metric: List[Union[int, float]], title: str, epoch: int, folder: str, ylabel: str, log_scale=False, no_xticks=False
):
    fig = plt.figure(figsize=(14, 6))
    plt.plot(metric)
    plt.title(title)
    if not no_xticks:
        plt.xticks(range(1, len(metric) + 1, 24), range(1, len(metric) // 24 + 1))
    plt.ylabel(ylabel)
    plt.xlabel("Days")
    if log_scale:
        plt.yscale("log")
    plt.tight_layout()
    fig.savefig(f"{folder}/{title} {epoch}.png")
    plt.close(fig)


def bar_metric(
    x_values: List[int],
    metric: List[int],
    title: str,
    epoch: int,
    folder: str,
    xlabel: str,
    ylabel: str,
    y_lim=None,
    xticks=None,
):
    fig = plt.figure(figsize=(14, 6))
    plt.bar(x_values, metric)
    plt.title(title)
    if y_lim:
        plt.ylim(y_lim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xticks is not None:
        plt.xticks(*xticks)
    plt.tight_layout()
    fig.savefig(f"{folder}/{title} {epoch}.png")
    plt.close(fig)


def plot_transaction_curve(cost: List[float], policy, epoch):
    folder = f"plots/raw_{policy}"
    length = len(cost)
    nc = EnergyCurve("./data/GR-data-11-20.csv", "eval")
    energy_demand = nc.get_y()[:length]
    buying = [[], [], []]
    nothing = [[], []]
    selling = [[], [], []]
    for (i, c), ed in zip(enumerate(cost), energy_demand):
        if c > 0:
            buying[0].append(i)
            buying[1].append(ed)
            buying[2].append(c ** 0.5)
        elif c == 0:
            nothing[0].append(i)
            nothing[1].append(ed)
        elif c < 0:
            selling[0].append(i)
            selling[1].append(ed)
            selling[2].append(abs(c) ** 0.5)

    plt.figure(figsize=(14, 6))
    plt.plot(energy_demand, label="Energy Demand Curve")
    plt.scatter(buying[0], buying[1], s=buying[2], label="Buying", c="g")
    plt.scatter(nothing[0], nothing[1], label="Nothing", c="darkgrey")
    plt.scatter(selling[0], selling[1], s=selling[2], label="Selling", c="r")
    plt.xticks(range(1, length + 1, 24), range(1, length // 24 + 1))
    plt.xlabel("Days")
    plt.ylabel("Energy Demand (Euro / MWh)")
    plt.legend()
    plt.title("Transactions")
    plt.tight_layout()
    plt.savefig(f"{folder}/Transactions {epoch}.png")
    plt.close()


def get_frequency(iteratable):
    counter = Counter(iteratable)
    x = []
    y = []
    for v, f in sorted(counter.items()):
        x.append(v)
        y.append(f)
    return x, y


def metrics_raw(metrics, epoch, policy):
    raw_folder = f"plots/raw_{policy}"
    plot_metric(metrics["cost"], "Cost", epoch, raw_folder, "Transaction Cost (Euro Cents)")
    plot_metric(
        metrics["cycle_degradation"], "Cycle degradation", epoch, raw_folder, "Cycle degradation cost (Euro Cents)"
    )
    plot_metric(metrics["age_degradation"], "Age degradation", epoch, raw_folder, "Age degradation cost (Euro Cents)")
    plot_metric(metrics["num_of_vehicles"], "Number of vehicles", epoch, raw_folder, "Number of vehicles")


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def metrics_moving_average(metrics, epoch, policy):
    avg_folder = f"plots/average_{policy}"
    window = 24
    plot_metric(moving_average(metrics["cost"], window), "Cost", epoch, avg_folder)
    plot_metric(moving_average(metrics["num_of_vehicles"], window), "Number of vehicles", epoch, avg_folder)
    plot_metric(moving_average(metrics["cycle_degradation"], window), "Cycle degradation", epoch, avg_folder)
    plot_metric(moving_average(metrics["age_degradation"], window), "Age degradation", epoch, avg_folder)


def metrics_frequency(metrics, epoch, policy):
    avg_folder = f"plots/average_{policy}"
    x_values, y_values = get_frequency(metrics["overcharged_time_per_car"])
    bar_metric(x_values, y_values, "Overcharged times", epoch, avg_folder, "Overcharged hours", "Number of vehicles")

    x_values, y_values = get_frequency(
        [int(c_rate * 100) for c_rate in metrics["c_rate_per_car"] if c_rate >= 0.01 or c_rate <= -0.01]
    )
    bar_metric(
        x_values,
        y_values,
        "C rate",
        epoch,
        avg_folder,
        "C rate",
        "Occurrences",
        xticks=(range(-100, 110, 10), map(lambda x: round(x, 1), np.arange(-1, 1.1, 0.1))),
    )


def metrics_simple(metrics, epoch, policy):
    avg_folder = f"plots/average_{policy}"
    total_sum = {
        "cost": 0.0,
        "cycle_degradation": 0.0,
        "age_degradation": 0.0,
    }
    epoch_length = len(metrics["cost"])
    for i in range(epoch_length):
        total_sum["cost"] += metrics["cost"][i]
        total_sum["cycle_degradation"] += metrics["cycle_degradation"][i]
        total_sum["age_degradation"] += metrics["age_degradation"][i]

    with open(f"{avg_folder}/total_cost_{epoch}.txt", "w") as f:
        f.write(f'Average Cost / hour: {total_sum["cost"] / len(metrics["cost"])}\n')
        f.write(
            f'Average Cycle Degradation / hour: {total_sum["cycle_degradation"] / len(metrics["cycle_degradation"])}\n'
        )
        f.write(f'Average Age Degradation / hour: {total_sum["age_degradation"] / len(metrics["age_degradation"])}\n')


def metrics_visualization(metrics: Dict[str, List[Union[int, float]]], epoch: int, policy: str):
    if epoch % 4 == 0:
        epoch = 4
    else:
        epoch %= 4
    plot_transaction_curve(metrics["cost"], policy, epoch)
    metrics_raw(metrics, epoch, policy)
    metrics_frequency(metrics, epoch, policy)
    metrics_simple(metrics, epoch, policy)
