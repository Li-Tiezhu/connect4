from stable_baselines3 import DQN
from connect import ConnectX
from stable_baselines3.common.env_checker import check_env


RED_PLAYER_VAL = -1
YELLOW_PLAYER_VAL = 1
GAME_STATE_NOT_ENDED = 2

env = ConnectX()
print(check_env(env))


# Model = DQN("MlpPolicy",
#             env,
#             verbose=1,
#             device='cuda',
#             tensorboard_log='./logs',
#             policy_kwargs={"net_arch": [700, 700, 700, 700, 700, 700]},
#             learning_rate=0.001,
#             batch_size=100,
#             buffer_size=50000,
#             learning_starts=600,
#             gamma=0.75
#             )
