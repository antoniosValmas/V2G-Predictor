from tf_agents.environments.py_environment import PyEnvironment


def compute_avg_return(environment: PyEnvironment, policy, num_episodes=10):
    total_return = 0.0
    step = 0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            step += 1
        total_return += episode_return

    avg_return = total_return / (num_episodes * step)
    return avg_return.numpy()[0]
