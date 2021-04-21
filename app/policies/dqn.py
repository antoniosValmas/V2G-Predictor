from __future__ import absolute_import, division, print_function
from app.policies.utils import compute_avg_return

from app.models.environment import V2GEnvironment

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common


class DQNPolicy:
    num_iterations = 4000  # @param {type:"integer"}

    initial_collect_steps = 100  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = 10000  # @param {type:"integer"}

    batch_size = 32  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 24  # @param {type:"integer"}

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 2400  # @param {type:"integer"}

    def __init__(self, train_env: V2GEnvironment, eval_env: V2GEnvironment):
        self.train_env = tf_py_environment.TFPyEnvironment(train_env)
        self.eval_env = tf_py_environment.TFPyEnvironment(eval_env)

        self.q_net = sequential.Sequential(
            [
                tf.keras.layers.Dense(
                    units=train_env.observation_spec().shape,
                    activation=tf.keras.activations.relu,
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(units=1024, activation=tf.keras.activations.relu),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(
                    units=train_env.action_spec().shape,
                    activation=tf.keras.activations.softmax,
                ),
            ]
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.agent = dqn_agent.DdqnAgent(
            time_step_spec=train_env.time_step_spec(),
            action_spec=train_env.action_spec(),
            q_network=self.q_net,
            optimizer=self.optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=self.train_step_counter,
        )

        self.agent.initialize()

        self.eval_policy = self.agent.policy
        self.collect_policy = self.agent.collect_policy

        self.random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=train_env.batch_size,
            max_length=self.replay_buffer_max_length,
        )

        replay_observer = [self.replay_buffer.add_batch]

        self.collect_op = dynamic_step_driver.DynamicStepDriver(
            train_env, self.agent.collect_policy, observers=replay_observer, num_steps=self.collect_steps_per_iteration
        ).run()

        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3, sample_batch_size=self.batch_size, num_steps=2
        ).prefetch(3)

        self.iterator = iter(self.dataset)

    def train(self):
        self.agent.train = common.function(self.agent.train)
        self.agent.train_step_counter.assign(0)
        avg_return = compute_avg_return(self.eval_env, self.agent.policy, self.num_eval_episodes)
        returns = [avg_return]

        for _ in range(self.num_iterations):

            # Sample a batch of data from the buffer and update the agent's network.
            experience, _ = next(self.iterator)
            train_loss = self.agent.train(experience).loss

            step = self.agent.train_step_counter.numpy()

            if step % self.log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss))

            if step % self.eval_interval == 0:
                avg_return = compute_avg_return(self.eval_env, self.agent.policy, self.num_eval_episodes)
                print('step = {0}: Average Return = {1}'.format(step, avg_return))
                returns.append(avg_return)
