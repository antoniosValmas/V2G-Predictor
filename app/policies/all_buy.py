import math
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories.policy_step import PolicyStep


class AllBuy:
    min_charge_idx = 28
    actions_length = 10

    def __init__(self, threshold):
        self.threshold = threshold

    def action(self, timestep: TimeStep) -> PolicyStep:
        observation = timestep.observation.numpy()[0]
        current_price = observation[3]
        good_price = current_price < self.threshold
        coefficient = 0.0

        if good_price:
            coefficient = 1 - current_price / self.threshold

        min_charge = observation[self.min_charge_idx]
        if min_charge > 0:
            needed_demand = min_charge / observation[self.min_charge_idx - 1]
            coefficient = max(coefficient, needed_demand)

        if (coefficient > 1):
            coefficient = 1

        return PolicyStep(action=10 + math.ceil(self.actions_length * coefficient))
