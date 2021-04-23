import app
from app.policies.dqn import DQNPolicy
from app.models.environment import V2GEnvironment

train_env = V2GEnvironment(100, './data/GR-data-11-20.csv', 'train')
eval_env = V2GEnvironment(100, './data/GR-data-11-20.csv', 'eval')

dqn = DQNPolicy(train_env, eval_env)
dqn.train()
