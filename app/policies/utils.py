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

    avg_return = total_return / (num_episodes * step)
    return avg_return.numpy()[0]


def plot_metric(metric: List[Union[int, float]], title: str, folder: str):
    fig = plt.figure(figsize=(14, 6))
    plt.plot(metric)
    plt.title(title)
    fig.savefig(f'{folder}/{title}.png')
    plt.close(fig)


def bar_metric(x_values: List[int], metric: List[int], title: str, folder: str):
    fig = plt.figure(figsize=(14, 6))
    plt.bar(x_values, metric)
    plt.title(title)
    fig.savefig(f'{folder}/{title}.png')
    plt.close(fig)


def get_frequency(iteratable):
    counter = Counter(iteratable)
    x = []
    y = []
    for v, f in sorted(counter.items()):
        x.append(v)
        y.append(f)
    return x, y


def metrics_raw(metrics, epoch):
    raw_folder = 'plots/metrics'
    plot_metric(metrics["cost"], 'Cost', raw_folder)
    plot_metric(metrics["unmet_demand"], 'Unmet demand', raw_folder)
    plot_metric(metrics["cycle_degradation"], 'Cycle degradation', raw_folder)
    plot_metric(metrics["age_degradation"], 'Age degradation', raw_folder)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def metrics_moving_average(metrics, epoch, policy):
    avg_folder = f'plots/average_{policy}'
    window = 24
    plot_metric(
        moving_average(metrics["cost"], window),
        f'Cost {epoch}',
        avg_folder
    )
    plot_metric(
        moving_average(metrics["unmet_demand"], window),
        f'Unmet demand {epoch}',
        avg_folder
    )
    plot_metric(
        moving_average(metrics["cycle_degradation"], window),
        f'Cycle degradation {epoch}',
        avg_folder
    )
    plot_metric(
        moving_average(metrics["age_degradation"], window),
        f'Age degradation {epoch}',
        avg_folder
    )


def metrics_frequency(metrics, epoch, policy):
    avg_folder = f'plots/average_{policy}'
    x_values, y_values = get_frequency(metrics["overcharged_time_per_car"])
    bar_metric(x_values, y_values, f"Overcharged time per car {epoch}", avg_folder)

    x_values, y_values = get_frequency([int(v * 100) for v in metrics["unmet_demand_per_car"]])
    bar_metric(x_values, y_values, f"Unmet demand per car {epoch}", avg_folder)


def metrics_visualization(metrics: Dict[str, List[Union[int, float]]], epoch: int, policy: str):
    metrics_moving_average(metrics, epoch, policy)
    metrics_frequency(metrics, epoch, policy)
