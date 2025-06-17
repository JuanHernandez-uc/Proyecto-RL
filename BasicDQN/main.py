import gymnasium as gym
from DQN import DQN
from stable_baselines3.common.monitor import Monitor

dqn_params = dict(
    learning_rate = 0.001,
    buffer_size = 1_000_000,
    learning_starts = 1_000,
    batch_size = 128,
    tau = 1,
    gamma = 0.99,
    train_freq = 16,
    gradient_steps = 4,
    target_update_interval = 1_000,
    exploration_fraction = 0.1,
    exploration_initial_eps = 0.1,
    exploration_final_eps = 0.1,
    max_grad_norm = 10,
    verbose = 1,
)

env = Monitor(gym.make("MountainCar-v0"))
my_dqn = DQN(env, **dqn_params)
my_dqn.learn(total_timesteps=300_000, log_interval=300)