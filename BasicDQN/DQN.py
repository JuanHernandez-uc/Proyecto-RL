import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import random
from torch.nn import functional as F
from collections import deque
from .MLP import MLP
from .buffer import FastReplayBuffer

class DQN:
    def __init__(self,
            env,
            learning_rate = 1e-3,
            buffer_size = 300000,
            learning_starts = 100,
            batch_size = 32,
            tau = 1.0,
            gamma = 0.99,
            train_freq = 4,
            gradient_steps = 1,
            target_update_interval = 10000,
            exploration_fraction = 0.1,
            exploration_initial_eps = 1.0,
            exploration_final_eps = 0.05,
            max_grad_norm = 10,
            verbose = 0
        ):
        
        ## Ambiente, dimensiones del estado y espacio de acciones
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ## Política online y política target
        self.policy = MLP(self.obs_dim, self.n_actions).to(self.device)
        self.target_policy = MLP(self.obs_dim, self.n_actions, orthogonal_init = False).to(self.device)
        self.target_policy.load_state_dict(self.policy.state_dict())

        ## Optimizador del MLP
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, betas=(0.9, 0.999))

        ## Replay buffer
        self.replay_buffer = FastReplayBuffer(buffer_size, self.obs_dim, self.n_actions, self.device)

        ## Definición de parámetros
        self.learning_starts = learning_starts
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.target_update_interval = target_update_interval
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.max_grad_norm = max_grad_norm
        self.verbose = verbose
        self.epsilon = self.exploration_initial_eps
        self.last_loss = None
        self.uses_monitor = False

        ## Cuántas veces se ha entrenado la red
        self.num_updates = 0
        
        ## Cuándo se debe sincronizar la red online con la target
        self._n_calls = 0

        ## Scheduler para ir bajando el epsilon. Parecido al de SB3
        self.eps_schedule = lambda progress: (
            self.exploration_final_eps
            + (self.exploration_initial_eps - self.exploration_final_eps) * progress
        )

    def learn(self, total_timesteps, log_interval=1000):
        self.learn_info = {
            "ep_count": 0,
            "episode_reward": 0,
            "start_time": time.perf_counter(),
            "ep_lens": deque(maxlen=100),
            "ep_rews": deque(maxlen=100)
        }

        state, _ = self.env.reset()
    
        for timestep in range(1, total_timesteps + 1):
            ## Progreso para calcular el decaimiento del epsilon
            progress = timestep / total_timesteps
            self.epsilon = self.eps_schedule(1.0 - progress)

            ## Tipico Q-learning
            action = self._select_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
        
            self.replay_buffer.add(state, action, reward, next_state, float(done))

            state = next_state

            self.learn_info["episode_reward"] += reward
    
            ## Si se ejecuta el ambiente con un monitor
            if "episode" in info:
                self.uses_monitor = True
                ep_len = info["episode"]["l"]
                ep_rew = info["episode"]["r"]
                self.learn_info["ep_lens"].append(ep_len)
                self.learn_info["ep_rews"].append(ep_rew)

            ## Cuándo se entrena la red
            if timestep > self.learning_starts and timestep % self.train_freq == 0:
                self._train_step()

            ## Sincronización de red target
            self._update_target_network()
            
            ## Fin del episodio
            if done:
                state, _ = self.env.reset()
                self.learn_info["ep_count"] += 1
                self.learn_info["episode_reward"] = 0

                if self.verbose and self.uses_monitor and self.learn_info["ep_count"] % log_interval == 0:
                    self.print_learn_info(timestep)

    ## Método para seleccionar acción con epsilon greedy
    def _select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy(state_tensor)

            ## Se elige la acción greedy
            return q_values.argmax(dim=1).item()

    ## Entrenamiento de la red
    def _train_step(self):
        ## Si no hay suficientes experiencias para entrenar en el buffer (caso borde)
        if len(self.replay_buffer) < self.batch_size:
            return

        self.policy.train()

        ## Típico entrenamiento de DL
        for _ in range(self.gradient_steps):
            ## Obtener batch
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)

            ## Calcular q values
            q_values = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                ## Inferir siguientes q values y calcular expected q
                next_q_values = self.target_policy(next_states).max(dim=1)[0]
                expected_q = rewards + self.gamma * next_q_values * (1 - dones)

            ## Pérdida de la red
            loss = F.smooth_l1_loss(q_values, expected_q)
            self.last_loss = loss.item()

            ## Optimización y back propagation
            self.optimizer.zero_grad()
            loss.backward()

            ## Clipeo del gradiente por si acaso
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

            self.optimizer.step()
            self.num_updates += 1

    def _update_target_network(self):
        if self.num_updates == 0:
            return
        
        self._n_calls += 1

        if self._n_calls % self.target_update_interval == 0:
            for target_param, policy_param in zip(self.target_policy.parameters(), self.policy.parameters()):
               ## target_param ← (1 - tau) * target_param + tau * policy_param
               target_param.data.mul_(1 - self.tau).add_(self.tau * policy_param.data)

    def print_learn_info(self, timestep):
        fps = int(timestep / (time.perf_counter() - self.learn_info["start_time"]))
        print("-" * 32)
        print(f"| rollout/            | {'':<6} |")
        print(f"|    ep_len_mean      | {np.mean(self.learn_info["ep_lens"]) if self.learn_info["ep_lens"] else float('nan'):<6.0f} |")
        print(f"|    ep_rew_mean      | {np.mean(self.learn_info["ep_rews"]) if self.learn_info["ep_rews"] else float('nan'):<6.0f} |")
        print(f"|    exploration_rate | {self.epsilon:<6.3f} |")
        print(f"| time/               | {'':<6} |")
        print(f"|    episodes         | {self.learn_info["ep_count"]:<6} | ")
        print(f"|    fps              | {fps:<6} |")
        print(f"|    time_elapsed     | {int(time.perf_counter() - self.learn_info["start_time"]):<6.0f} |")
        print(f"|    total_timesteps  | {timestep:<6} |")
        print(f"| train/              | {'':<6} |")
        print(f"|    learning_rate    | {self.learning_rate:<6.3g} |")
        print(f"|    loss             | {self.last_loss if self.last_loss is not None else float("nan"):<6.4f} |")
        print(f"|    n_updates        | {self.num_updates:<6} |")
        print("-" * 32)