from app.policies.utils import compute_avg_return, metrics_visualization
import app
from app.policies.dqn import DQNPolicy
from app.models.environment import V2GEnvironment

train_env = V2GEnvironment(1000, './data/GR-data-11-20.csv', 'train')
eval_env = V2GEnvironment(1000, './data/GR-data-11-20.csv', 'eval')

dqn = DQNPolicy(train_env, eval_env)
compute_avg_return(dqn.eval_env, dqn.agent.policy, 1)

metrics_visualization(eval_env.get_metrics(), 0)
