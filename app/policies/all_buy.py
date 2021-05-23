import math
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories.policy_step import PolicyStep


class AllBuy:
    actions_length = 21

    def __init__(self, threshold):
        self.threshold = threshold

    def action(self, timestep: TimeStep) -> PolicyStep:
        observation = timestep.observation.numpy()[0]
        current_price = observation[3]
        max_coefficient, threshold_coefficient, min_coefficient = observation[:3]
        coefficient_step = (max_coefficient - min_coefficient) / (self.actions_length - 1)
        if coefficient_step == 0:
            # print(max_coefficient, threshold_coefficient, min_coefficient, 20)
            return PolicyStep(action=self.actions_length - 1)

        threshold_offset = math.ceil((threshold_coefficient - min_coefficient) / coefficient_step)
        good_price = current_price < self.threshold
        coefficient = 0.0

        if good_price:
            coefficient = 1 - current_price / self.threshold

        # print(
        #     max_coefficient,
        #     threshold_coefficient,
        #     min_coefficient,
        #     threshold_offset,
        #     threshold_offset + math.floor((self.actions_length - threshold_offset - 1) * coefficient),
        # )
        return PolicyStep(
            action=threshold_offset + math.floor((self.actions_length - threshold_offset - 1) * coefficient)
        )
