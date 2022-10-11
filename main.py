from stable_baselines3 import DQN
from connect import ConnectX
import numpy as np
from stable_baselines3.common.env_checker import check_env

RED_PLAYER_VAL = -1
YELLOW_PLAYER_VAL = 1
GAME_STATE_NOT_ENDED = 2

env = ConnectX()

Model = DQN("MlpPolicy",
            env,
            verbose=1,
            device='cuda',
            tensorboard_log='./logs',
            policy_kwargs={"net_arch": [700, 700, 700, 700, 700, 700]},
            learning_rate=0.001,
            batch_size=100,
            buffer_size=50000,
            learning_starts=600,
            exploration_fraction=0.05,
            gamma=0.75,
            )

Model.learn(total_timesteps=int(1e6), tb_log_name='DQN_net_arch_learning_buffer')

Model.save("dqn_sumo_net_arch_learning_buffer")


# Model = DQN.load('dqn_sumo_net_arch_learning_buffer', env=env)
#
#
# def my_agent(observation, configuration):
#     print(observation)
#     obs = np.array(observation['board'])
#     obs = np.expand_dims(obs, axis=0)
#     obs = np.expand_dims(obs, axis=0)
#     obs = obs.astype(np.uint8)
#     action, _states = Model.predict(obs, deterministic=True)
#     if type(action) is np.ndarray:
#         action = int(action)
#     print("my action:", action, ", action type:", type(action))
#     return action
#
#
# env.reset()
# # Play as the first agent against default "random" agent.
# env.run([my_agent, "random"])
# env.render(width=500, height=450)
