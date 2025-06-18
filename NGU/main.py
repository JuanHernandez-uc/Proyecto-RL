import gymnasium as gym
from DQN import DQN
from stable_baselines3.common.monitor import Monitor
import copy

# Parámetros comunes
base_params = dict(
    learning_rate = 0.001,
    buffer_size = 1_000_000,
    learning_starts = 1_000,
    batch_size = 128,
    tau = 1.0,
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

# Copias con y sin NGU
ngu_params = copy.deepcopy(base_params)
ngu_params["enable_ngu"] = True
ngu_params["intrinsic_coef"] = 0.3

basic_params = copy.deepcopy(base_params)
basic_params["enable_ngu"] = False  # En caso de que BasicDQN no lo reconozca, se ignora

# Entornos distintos (con monitores)
env_ngu = Monitor(gym.make("MountainCar-v0"))
env_basic = Monitor(gym.make("MountainCar-v0"))

# Agentes
agent_ngu = DQN(env_ngu, **ngu_params)
agent_basic = DQN(env_basic, **basic_params)

# Entrenamiento (puedes hacerlo en paralelo si usas multiprocessing o Jupyter)
agent_ngu.learn(total_timesteps=300_000, log_interval=300)
agent_basic.learn(total_timesteps=300_000, log_interval=300)

import matplotlib.pyplot as plt

# Extraer listas crudas
rew_basic = list(agent_basic.learn_info["ep_rews"])
rew_ngu = list(agent_ngu.learn_info["ep_rews"])
len_basic = list(agent_basic.learn_info["ep_lens"])
len_ngu = list(agent_ngu.learn_info["ep_lens"])

# Crear figura con dos gráficos lado a lado
plt.figure(figsize=(12, 5))

# Gráfico de recompensas por episodio
plt.subplot(1, 2, 1)
plt.plot(rew_basic, label="BasicDQN")
plt.plot(rew_ngu, label="NGU_DQN")
plt.title("Recompensa por Episodio (crudo)")
plt.xlabel("Episodio")
plt.ylabel("Recompensa")
plt.legend()
plt.grid(True)

# Gráfico de longitud de episodios
plt.subplot(1, 2, 2)
plt.plot(len_basic, label="BasicDQN")
plt.plot(len_ngu, label="NGU_DQN")
plt.title("Duración del Episodio (crudo)")
plt.xlabel("Episodio")
plt.ylabel("Longitud")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
