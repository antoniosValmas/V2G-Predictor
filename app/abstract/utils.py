from tf_agents.environments.py_environment import PyEnvironment


def compute_avg_return(environment: PyEnvironment, policy, num_episodes=10):
    total_return = 0.0
    step = 0

    for _ in range(num_episodes):

        # Reset the environment
        time_step = environment.reset()
        # Initialize the episode return
        episode_return = 0.0

        # While the current state is not a terminal state
        while not time_step.is_last():
            # Use policy to get the next action
            action_step = policy.action(time_step)
            # Apply action on the environment to get next state
            time_step = environment.step(action_step.action)
            # Add reward on the episode return
            episode_return += time_step.reward
            # Increase step counter
            step += 1
        # Add episode return on total_return counter
        total_return += episode_return

    # Calculate average return
    avg_return = total_return / step
    # Unpack value
    return avg_return.numpy()[0]
