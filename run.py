import app
from app.policies.dqn import DQNPolicy
from app.models.environment import V2GEnvironment
from app.utils import create_vehicle_distribution

vehicles = create_vehicle_distribution()

train_env = V2GEnvironment(100, './data/GR-data-11-20.csv', 'train')
eval_env = V2GEnvironment(200, './data/GR-data-11-20.csv', 'eval', vehicles)

dqn = DQNPolicy(train_env, eval_env)
dqn.train()

# metrics_visualization(train_env.get_metrics())
