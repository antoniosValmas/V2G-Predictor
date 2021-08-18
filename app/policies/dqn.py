from __future__ import absolute_import, division, print_function
from app.policies.dummy_v2g import DummyV2G

from app.policies.utils import compute_avg_return, metrics_visualization, moving_average, plot_metric
from app.models.environment import V2GEnvironment

import sys
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy, policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.trajectories import trajectory


class DQNPolicy:
    num_iterations = 24 * 30 * 50  # @param {type:"integer"}

    initial_collect_steps = 24 * 5  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = 1000000  # @param {type:"integer"}

    batch_size = 32  # @param {type:"integer"}
    learning_rate = 1e-4  # @param {type:"number"}
    log_interval = 24  # @param {type:"integer"}

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 24 * 30 * 2  # @param {type:"integer"}

    train_dir = "checkpoints"

    def __init__(self, train_env: V2GEnvironment, eval_env: V2GEnvironment):
        self.raw_train_env = train_env
        self.raw_eval_env = eval_env
        self.train_env = tf_py_environment.TFPyEnvironment(train_env)
        self.eval_env = tf_py_environment.TFPyEnvironment(eval_env)

        self.global_step = tf.Variable(0)
        action_tensor_spec = tensor_spec.from_spec(train_env.action_spec())
        num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

        self.q_net = sequential.Sequential(
            [
                layers.Dense(units=33, activation="elu"),
                layers.Dense(units=128, activation="elu"),
                layers.BatchNormalization(),
                # layers.Dropout(0.4),
                layers.Dense(
                    num_actions,
                    activation=None,
                ),
            ]
        )

        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.agent = dqn_agent.DdqnAgent(
            time_step_spec=self.train_env.time_step_spec(),
            action_spec=self.train_env.action_spec(),
            q_network=self.q_net,
            optimizer=self.optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=self.global_step,
            target_update_tau=0.001,
            target_update_period=1,
        )

        self.agent.initialize()

        self.eval_policy = self.agent.policy
        self.collect_policy = self.agent.collect_policy

        self.random_policy = random_tf_policy.RandomTFPolicy(
            self.train_env.time_step_spec(), self.train_env.action_spec()
        )

        self.policy_saver = policy_saver.PolicySaver(self.agent.policy, self.train_env.batch_size)

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=self.replay_buffer_max_length,
        )

        self.train_checkpointer = common.Checkpointer(
            ckpt_dir=self.train_dir,
            max_to_keep=5,
            agent=self.agent,
            policy=self.agent.policy,
            replay_buffer=self.replay_buffer,
            global_step=self.global_step,
        )

        if self.train_checkpointer.checkpoint_exists:
            print("Found Train Checkpoint")

        self.train_checkpointer.initialize_or_restore()

        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3, sample_batch_size=self.batch_size, num_steps=2
        ).prefetch(3)

        self.iterator = iter(self.dataset)

    def train(self, load_policy=None):
        self.train_env.reset()
        self.eval_env.reset()
        print("Collect Step")

        dummy_v2g = DummyV2G(0.5)
        self.collect_data(self.train_env, dummy_v2g, self.initial_collect_steps)
        self.agent.train = common.function(self.agent.train)
        self.raw_eval_env.reset_metrics()
        returns = []
        loss = []

        for i in range(self.num_iterations):

            try:
                self.collect_data(self.train_env, self.agent.policy, 1)

                # Sample a batch of data from the buffer and update the agent's network.
                experience, _ = next(self.iterator)
                train_loss = self.agent.train(experience).loss
                loss.append(train_loss)

                step = self.agent.train_step_counter.numpy()

                if step % self.log_interval == 0:
                    remainder = step % self.eval_interval
                    percentage = ((remainder if remainder != 0 else self.eval_interval) * 70) // self.eval_interval
                    sys.stdout.write("\r")
                    sys.stdout.write("\033[K")
                    sys.stdout.write(f'[{"=" * percentage + " " * (70 - percentage)}] loss: {train_loss} ')
                    sys.stdout.flush()

                if step % self.eval_interval == 0:
                    avg_return = compute_avg_return(self.eval_env, self.agent.policy, self.num_eval_episodes)
                    epoch = (i + 1) // self.eval_interval
                    total_epochs = self.num_iterations // self.eval_interval
                    print(
                        "Epoch: {0}/{1} step = {2}: Average Return = {3}".format(epoch, total_epochs, step, avg_return)
                    )
                    returns.append(avg_return)

                    metrics_visualization(self.raw_eval_env.get_metrics(), (i + 1) // self.eval_interval, "dqn")
                    self.raw_eval_env.hard_reset()

                    # self.collect_data(self.train_env, self.random_policy, self.initial_collect_steps)

            except ValueError as e:
                print(self.train_env.current_time_step())
                raise e

        self.train_checkpointer.save(global_step=self.global_step.numpy())
        plot_metric(returns, "Average Returns", 0, "plots/raw_dqn", "Average Return", no_xticks=True)
        plot_metric(
            moving_average(loss, 240), "Training Loss", 0, "plots/raw_dqn", "Loss", log_scale=True, no_xticks=True
        )

    def collect_step(self, environment: tf_py_environment.TFPyEnvironment, policy):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        if next_time_step.is_last():
            environment.reset()
        # Add trajectory to the replay buffer
        self.replay_buffer.add_batch(traj)

    def collect_data(self, env, policy, steps):
        for _ in range(steps):
            self.collect_step(env, policy)
