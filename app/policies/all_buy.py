import math
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories.policy_step import PolicyStep


class AllBuy:
    min_charge_idx = 25
    actions_length = 10

    def __init__(self, threshold):
        self.threshold = threshold

    def action(self, timestep: TimeStep) -> PolicyStep:
        observation = timestep.observation.numpy()[0]
        good_price = observation[0] < self.threshold
        coefficient = 0.0

        if good_price:
            coefficient = 1 - observation[0] / self.threshold

        if observation[self.min_charge_idx] > 0:
            needed_demand = observation[self.min_charge_idx] / observation[self.min_charge_idx - 1]
            coefficient = max(coefficient, needed_demand)

        if (coefficient > 1):
            coefficient = 1

        return PolicyStep(action=10 + math.ceil(self.actions_length * coefficient))
