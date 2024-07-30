"""
2024.7.20
ToothlessOS
MARL testcase with skrl
Documentation references:
- https://skrl.readthedocs.io/en/latest/api/multi_agents/mappo.html
- https://pettingzoo.farama.org/environments/sisl/multiwalker/
"""

# Envs
from pettingzoo.sisl import multiwalker_v9
from skrl.envs.wrappers.torch import wrap_env
# Memories
from skrl.memories.torch import RandomMemory
# Import MAPPO agent and default configuration
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG

# PettingZoo env setup
env = multiwalker_v9.parallel_env()
env = wrap_env(env)

# Memory setup
memories = RandomMemory(memory_size=1024, num_envs=env.num_envs, device=env.device)

# instantiate the agent's models
models = {}
for agent_name in env.possible_agents:
    models[agent_name] = {}
    models[agent_name]["policy"] = ...
    models[agent_name]["value"] = ...  # only required during training

# adjust some configuration if necessary (hyperparameters, etc.)
cfg_agent = MAPPO_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memories <memories>)
agent = MAPPO(possible_agents=env.possible_agents,
              models=models,
              memory=memories,  # only required during training
              cfg=cfg_agent,
              observation_spaces=env.observation_spaces,
              action_spaces=env.action_spaces,
              device=env.device,
              shared_observation_spaces=env.shared_observation_spaces)