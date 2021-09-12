from __future__ import absolute_import, division, print_function

import tensorflow as tf

# !!! Change this, if folder structure is different !!!
from app.abstract.utils import compute_avg_return

from tensorflow import keras
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.networks import sequential
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory

import sys


class DDQNPolicy:
    """
    An abstract class that initializes the DDQN agent, provided by tesnsorflow, along with some other helpful tools

    ### Constructor arguments
    - `train_env`: The environment to be used for training.
    - `eval_env`: The environment to be used for validation.
    - `q_net`: The neural network to be used for the DDQN agent.

    Both environment instances must comply with the PyEnvironment structure\
    (https://www.tensorflow.org/agents/tutorials/2_environments_tutorial).

    The neural network must be a sequential model https://keras.io/guides/sequential_model/).

    The variables are as follows (these can be overridden by manually changing their values):
    - `replay_buffer_max_length`: The maximum size of the replay buffer
    - `batch_size`: The batch size for the data
    - `learning_rate`: The learning rate for the optimizer
    - `train_dir`: The directory in which the checkpoints are saved

    The instances create / provided by the class are:
    - `raw_train_env`: The original test pyEnvironment instance
    - `raw_eval_env`: The original eval pyEnvironment instance
    - `train_env`: The transformed test tensor environment instance
    - `eval_env`: The transformed eval tensor environment instance
    - `global_step`: A global step counter used to count the total step the algorithm has made on the train environment
    - `agent`: The active instance of the DDQN agent
    - `eval_policy`: The evaluation policy of the agent
    - `collect_policy`: The collect policy of the agent
    - `replay_buffer`: The instance of the replay buffer
    - `train_checkpointer`: The instance of the checkpointer
    - `iterator`: The iterator instance of the replay buffer
    """

    # Initialization variables
    replay_buffer_max_length = 1000000  # @param {type:"integer"}
    batch_size = 32  # @param {type:"integer"}
    learning_rate = 1e-4  # @param {type:"number"}
    train_dir = "checkpoints"  # the checkpoint folder

    def __init__(self, train_env: PyEnvironment, eval_env: PyEnvironment, q_net: sequential.Sequential):
        self.raw_train_env = train_env
        self.raw_eval_env = eval_env

        # Transform py environment to tensor environments to increase efficiency
        self.train_env = tf_py_environment.TFPyEnvironment(train_env)
        self.eval_env = tf_py_environment.TFPyEnvironment(eval_env)

        # Set global step counter to 0 (this will be overwritten by reload of the checkpoint)
        self.global_step = tf.Variable(0)

        # Create a DDQN agent
        self.agent = dqn_agent.DdqnAgent(
            time_step_spec=self.train_env.time_step_spec(),
            action_spec=self.train_env.action_spec(),
            q_network=q_net,
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=self.global_step,
            target_update_tau=0.001,
            target_update_period=1,
        )
        self.agent.initialize()

        # Copy eval and collect policies
        self.eval_policy = self.agent.policy
        self.collect_policy = self.agent.collect_policy

        # Create a replay buffer
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=self.replay_buffer_max_length,
        )

        # Create a checkpointer instance
        # This is used to save the current state of the policy, agent and replay buffer
        self.train_checkpointer = common.Checkpointer(
            ckpt_dir=self.train_dir,
            max_to_keep=5,
            agent=self.agent,
            policy=self.agent.policy,
            replay_buffer=self.replay_buffer,
            global_step=self.global_step,
        )

        # If a checkpoint already exists in the "train_dir" folder, output a message informing the user
        if self.train_checkpointer.checkpoint_exists:
            print("Checkpoint found. Continuing from checkpoint...")

        # This will restore the agent, policy and replay buffer from the checkpoint, if it exists
        # Or it will just initialize everything to their initial setup
        self.train_checkpointer.initialize_or_restore()

        # Create a dataset and an iterator instance, this helps to create batches of data in training
        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3, sample_batch_size=self.batch_size, num_steps=2
        ).prefetch(3)
        self.iterator = iter(dataset)

    # Train variables
    num_iterations = 24 * 30 * 200  # @param {type:"integer"}

    initial_collect_steps = 24 * 5  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = 1000000  # @param {type:"integer"}

    batch_size = 32  # @param {type:"integer"}
    learning_rate = 1e-4  # @param {type:"number"}
    log_interval = 24  # @param {type:"integer"}

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 24 * 30 * 8  # @param {type:"integer"}

    progress_bar_length = 70

    def train(self):
        """
        A function to train the policy.

        - Uses a random policy to collect the initial steps to populate the replay buffer
        - It logs the progress using an ASCII progress bar.
        - Outputs the average return as a validation metric.
        - After finishing the training (num_iterations), it saves the policy, replay buffer
            and other useful variables, for re-use.
        """
        # Reset both environments
        self.train_env.reset()
        self.eval_env.reset()

        print("Collect Step")
        # Create a random policy to sample some data from the environment to populate the replay buffer
        random_policy = random_tf_policy.RandomTFPolicy(self.train_env.time_step_spec(), self.train_env.action_spec())
        self.collect_data(self.train_env, random_policy, self.initial_collect_steps)
        # Wrap the train function to speed it up
        self.agent.train = common.function(self.agent.train)

        returns = []
        loss = []

        for i in range(self.num_iterations):

            try:
                # Execute one step on the train_env and save it to the replay buffer
                self.collect_data(self.train_env, self.agent.policy, 1)

                # Sample a batch of data from the buffer and update the agent's network.
                experience, _ = next(self.iterator)
                train_loss = self.agent.train(experience).loss
                loss.append(train_loss)

                # Cast train_step counter into an numpy int to more easily manipulate it
                step = self.agent.train_step_counter.numpy()

                # Log interval
                if step % self.log_interval == 0:
                    remainder = step % self.eval_interval
                    # Calculate the percentage of the progress of the current epoch
                    percentage = (
                        (remainder if remainder != 0 else self.eval_interval) * self.progress_bar_length
                    ) // self.eval_interval

                    # Print a progress bar
                    sys.stdout.write("\r")
                    sys.stdout.write("\033[K")
                    sys.stdout.write(
                        f'[{"=" * percentage + " " * (self.progress_bar_length - percentage)}] loss: {train_loss} '
                    )
                    sys.stdout.flush()

                # Validation step
                if step % self.eval_interval == 0:
                    # Calculate the average return of the current trained policy
                    avg_return = compute_avg_return(self.eval_env, self.agent.policy, self.num_eval_episodes)
                    # Log validation average return next to progress bar
                    epoch = (i + 1) // self.eval_interval
                    total_epochs = self.num_iterations // self.eval_interval
                    print(
                        "Epoch: {0}/{1} step = {2}: Average Return = {3}".format(epoch, total_epochs, step, avg_return)
                    )
                    returns.append(avg_return)

            except ValueError as e:
                print("An exception has occurred while trying to train the agent")
                raise e

        # Save the progress of the policy and the replay buffer
        self.train_checkpointer.save(global_step=self.global_step.numpy())

    def collect_step(self, environment: tf_py_environment.TFPyEnvironment, policy):
        """
        Make a step based on a given policy, create the trajectory and save it to the replay buffer

        ### Arguments:
        - `environment`: The environment on which the action will take place
        - `policy`: The policy, which will provide the action, based on the previous state
        """

        # Get current time step
        time_step = environment.current_time_step()
        # Output action based on the policy
        action_step = policy.action(time_step)
        # Execute the action on the environment
        next_time_step = environment.step(action_step.action)
        # Create a trajectory (prev_state => action => next_state)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        # If next state is a terminal state then reset environment
        if next_time_step.is_last():
            environment.reset()
        # Add trajectory to the replay buffer
        self.replay_buffer.add_batch(traj)

    def collect_data(self, env, policy, steps):
        """
        Collect a certain amount of steps using the `collect_step` function

        ### Arguments:
        - `environment`: The environment on which the action will take place
        - `policy`: The policy, which will provide the action, based on the previous state
        - `steps`: The amount of steps to collect
        """
        # Collect as many steps as the counter suggests
        for _ in range(steps):
            self.collect_step(env, policy)
