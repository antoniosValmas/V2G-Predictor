from app.utils import create_vehicle_distribution
from tf_agents.environments import tf_py_environment
import app
from app.policies.dummy_v2g import DummyV2G
from app.policies.smart_charger import SmartCharger
from app.policies.utils import compute_avg_return, metrics_visualization
from app.policies.dqn import DQNPolicy
from app.models.environment import V2GEnvironment

train_env = V2GEnvironment(300, './data/GR-data-new.csv', 'train')
vehicles = create_vehicle_distribution(6552)

# DQN
eval_env = V2GEnvironment(300, './data/GR-data-new.csv', 'eval', vehicles)

dqn = DQNPolicy(train_env, eval_env)
compute_avg_return(dqn.eval_env, dqn.agent.policy, 273)

metrics_visualization(eval_env.get_metrics(), 0, 'dqn')

# Dummy V2G
eval_env = V2GEnvironment(300, './data/GR-data-new.csv', 'eval', vehicles)

tensor_eval_env = tf_py_environment.TFPyEnvironment(eval_env)

dummy_v2g = DummyV2G(0.5)
compute_avg_return(tensor_eval_env, dummy_v2g, 273)

metrics_visualization(eval_env.get_metrics(), 0, 'dummy_v2g')

# Smart Charger
eval_env = V2GEnvironment(300, './data/GR-data-new.csv', 'eval', vehicles)

tensor_eval_env = tf_py_environment.TFPyEnvironment(eval_env)

smart_charger = SmartCharger(0.5)
compute_avg_return(tensor_eval_env, smart_charger, 273)

metrics_visualization(eval_env.get_metrics(), 0, 'smart_charger')
