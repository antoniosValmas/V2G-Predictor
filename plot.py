from typing import List, Union

from app.policies.utils import get_frequency
from app.models.energy import EnergyCurve
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
import pickle
import numpy as np


def plot_metric(
    metric: List[Union[int, float]],
    title: str,
    ylabel: str,
    axes: Axes,
):
    axes.plot(metric)
    axes.set_title(title)
    axes.set_xticks(range(0, len(metric)))
    axes.set_ylabel(ylabel)
    axes.grid(True, "major", "both")


def bar_metric(
    x_values: List[int],
    metric: List[int],
    title: str,
    ylabel: str,
    axes: Axes,
    xticks=None,
):
    axes.bar(x_values, metric)
    axes.set_title(title)
    axes.set_ylabel(ylabel)
    if xticks is not None:
        plt.sca(axes)
        plt.xticks(*xticks)


def plot_transaction_curve(cost: List[float], policy: str, energy_demand: List[float], axes: Axes):
    length = len(cost)
    energy_demand = energy_demand[:length]
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

    axes.plot(energy_demand, label="Energy Demand Curve")
    axes.scatter(buying[0], buying[1], s=buying[2], label="Buying", c="g")
    axes.scatter(nothing[0], nothing[1], label="Nothing", c="darkgrey")
    axes.scatter(selling[0], selling[1], s=selling[2], label="Selling", c="r")
    axes.set_xticks(range(0, length))
    axes.set_ylabel("Energy Demand (Euro / MWh)")
    axes.set_title(policy)
    axes.grid(True, "major", "both")


cFig, cAx = plt.subplots(3, 1, sharex="all", figsize=(10, 12))
cDFig, cDAx = plt.subplots(3, 1, sharex="all", figsize=(10, 12))
aDFig, aDAx = plt.subplots(3, 1, sharex="all", figsize=(10, 12))
cRFig, cRAx = plt.subplots(3, 1, sharex="all", figsize=(12, 14.4))
oFig, oAx = plt.subplots(3, 1, sharex="all", figsize=(10, 12))
tFig, tAx = plt.subplots(3, 1, sharex="all", figsize=(10, 12))

for i, policy in enumerate(["Smart Charger", "V2G", "Optimized V2G"]):

    metrics = pickle.load(open(f"./data/metrics_{policy}", "rb"))

    nc = EnergyCurve("./data/GR-data-new.csv", "eval")
    plot_transaction_curve(metrics["cost"][24 * 15:24 * 16], policy, nc.get_y()[24 * 15:24 * 16], tAx[i])
    day16metrics = {
        "cost": metrics["cost"][24 * 15:24 * 16],
        "cycle_degradation": metrics["cycle_degradation"][24 * 15:24 * 16],
        "age_degradation": metrics["age_degradation"][24 * 15:24 * 16],
        "num_of_vehicles": metrics["num_of_vehicles"][24 * 15:24 * 16],
    }

    plot_metric(day16metrics["cost"], policy, "Transaction Cost (Euro Cents)", cAx[i])
    plot_metric(day16metrics["cycle_degradation"], policy, "Cycle degradation cost (Euro Cents)", cDAx[i])
    plot_metric(day16metrics["age_degradation"], policy, "Age degradation cost (Euro Cents)", aDAx[i])

    x_values, y_values = get_frequency(metrics["overcharged_time_per_car"])
    bar_metric(x_values, y_values, policy, "Number of vehicles", oAx[i])

    x_values, y_values = get_frequency(
        [int(c_rate * 100) for c_rate in metrics["c_rate_per_car"] if c_rate >= 0.01 or c_rate <= -0.01]
    )
    bar_metric(
        x_values,
        y_values,
        policy,
        "Occurrences",
        cRAx[i],
        xticks=(range(-100, 110, 10), list(map(lambda x: round(x, 1), np.arange(-1, 1.1, 0.1)))),
    )


mid = (cFig.subplotpars.right + cFig.subplotpars.left)/2
cAx[2].set_xlabel("Hour of the day")
cFig.suptitle("Cost curve", x=mid)
cFig.tight_layout()
cFig.savefig("plots/combined/Cost.png")


mid = (cFig.subplotpars.right + cFig.subplotpars.left)/2
cAx[2].set_xlabel("Hour of the day")
cFig.suptitle("Cost curve", x=mid)
cFig.tight_layout()
cFig.savefig("plots/combined/Cost.png")

mid = (cDFig.subplotpars.right + cDFig.subplotpars.left)/2
cDAx[2].set_xlabel("Hour of the day")
cDFig.suptitle("Cycle battery degradation curve", x=mid)
cDFig.tight_layout()
cDFig.savefig("plots/combined/Cycle_degradation.png")

mid = (aDFig.subplotpars.right + aDFig.subplotpars.left)/2
aDAx[2].set_xlabel("Hour of the day")
aDFig.suptitle("Age battery degradation curve", x=mid)
aDFig.tight_layout()
aDFig.savefig("plots/combined/Age_degradation.png")

mid = (cRFig.subplotpars.right + cRFig.subplotpars.left)/2
cRAx[2].set_xlabel("C rate")
cRFig.suptitle("C rate frequency plot", x=mid)
cRFig.tight_layout(pad=2)
cRFig.savefig("plots/combined/Crate.png")

mid = (tFig.subplotpars.right + tFig.subplotpars.left)/2
tAx[2].set_xlabel("Hour of the day")
tAx[0].legend()
tFig.suptitle("Transactions plot", x=mid)
tFig.tight_layout()
tFig.savefig("plots/combined/transactions.png")

plt.close(cFig)
plt.close(cDFig)
plt.close(aDFig)
plt.close(cRFig)
plt.close(oFig)
