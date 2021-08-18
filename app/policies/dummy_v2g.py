import numpy as np
import math
import tensorflow as tf
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories.policy_step import PolicyStep


class DummyV2G:
    actions_length = 21

    def __init__(self, threshold):
        self.threshold = threshold

    def action(self, timestep: TimeStep) -> PolicyStep:
        observation = timestep.observation.numpy()[0]
        current_price = observation[3]
        max_coefficient, threshold_coefficient, min_coefficient = observation[:3]
        coefficient_step = (max_coefficient - min_coefficient) / (self.actions_length - 1)
        # print(f"Price: {current_price}, Obs coeffs {max_coefficient, threshold_coefficient, min_coefficient}", end="")
        if coefficient_step == 0:
            return PolicyStep(action=tf.constant(np.array([self.actions_length - 1], dtype=np.int32)))

        good_price = current_price < self.threshold
        coefficient = threshold_coefficient

        if good_price:
            coefficient = (threshold_coefficient - max_coefficient) * (current_price / self.threshold) + max_coefficient
        else:
            coefficient = min_coefficient - (min_coefficient - threshold_coefficient) * math.e ** (
                1 - current_price / self.threshold
            )

        # print(f", coefficient: {coefficient}")

        return PolicyStep(
            action=tf.constant(np.array([round((coefficient - min_coefficient) / coefficient_step)], dtype=np.int32))
        )


# PolicyStep(action=<tf.Tensor: shape=(1,), dtype=int32, numpy=array([3], dtype=int32)>, state=(), info=())
