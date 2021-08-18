from app.policies.utils import metrics_raw, plot_transaction_curve
from app.models.energy import EnergyCurve
import pickle


for policy in ["dqn", "dummy_v2g", "smart_charger"]:
    metrics = pickle.load(open(f"./data/metrics_{policy}", "rb"))

    nc = EnergyCurve("./data/GR-data-new.csv", "eval")
    plot_transaction_curve(metrics["cost"][24*15:24*16], policy, 5, nc.get_y()[24*15:24*16])
    day16metrics = {
        "cost": metrics["cost"][24*15:24*16],
        "cycle_degradation": metrics["cycle_degradation"][24*15:24*16],
        "age_degradation": metrics["age_degradation"][24*15:24*16],
        "num_of_vehicles": metrics["num_of_vehicles"][24*15:24*16]
    }
    metrics_raw(day16metrics, 5, policy)
