from app.utils import create_vehicle_distribution
from tf_agents.environments import tf_py_environment
import app
from app.policies.all_buy import AllBuy
from app.policies.utils import compute_avg_return, metrics_visualization
from app.policies.dqn import DQNPolicy
from app.models.environment import V2GEnvironment

train_env = V2GEnvironment(300, './data/GR-data-11-20.csv', 'train')
vehicles = create_vehicle_distribution()

# DQN
eval_env = V2GEnvironment(300, './data/GR-data-11-20.csv', 'eval', vehicles)

dqn = DQNPolicy(train_env, eval_env)
compute_avg_return(dqn.eval_env, dqn.agent.policy, 30 * 6)

metrics_visualization(eval_env.get_metrics(), 0, 'dqn')

# All Buy
eval_env = V2GEnvironment(300, './data/GR-data-11-20.csv', 'eval', vehicles)

tensor_eval_env = tf_py_environment.TFPyEnvironment(eval_env)

all_buy = AllBuy(0.5)
compute_avg_return(tensor_eval_env, all_buy, 30 * 6)

metrics_visualization(eval_env.get_metrics(), 0, 'all_buy')
