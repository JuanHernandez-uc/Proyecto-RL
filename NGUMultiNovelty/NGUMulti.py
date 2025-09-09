import time
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import imageio
from PIL import Image
from collections import deque
from .contrastive import contrastive_multiview_loss
from .cosine_registry import CosineRegistry

class NGUMultiAgent:
    def __init__(self, env, agent_dict, total_timesteps, log_interval=1000):
        self.env = env
        self.agent_dict = agent_dict
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval

        self.ep_count = 0
        self.ep_lens = deque(maxlen=100)
        self.ep_rews = {agent: deque(maxlen=100) for agent in agent_dict}
        self.start_time = time.perf_counter()
        self.episode_rewards_log = []
        self.last_logged_ep = -1

        self.cosreg = CosineRegistry(
            sim_thresh=0.85,      # umbral de similitud coseno (ajusta 0.85-0.95)
            tau_mode="all",         # "any" | "k" | "all"
            k=3,                  # agentes necesarios para considerar "no-novedoso"
            n_agents=len(agent_dict)
        )

        self.alpha_contrast = 0.02   # 0.02–0.1 típico
        self.temp_contrast = 0.2

    def learn(self):
        observations, _ = self.env.reset()
        episode_reward = {agent: 0 for agent in self.agent_dict}
        ep_len = 0
        t = 0

        while t < self.total_timesteps:
            actions = {}
            for agent in self.env.agents:
                if agent in self.agent_dict:
                    dqn = self.agent_dict[agent]
                    dqn._update_epsilon(t, self.total_timesteps)
                    actions[agent] = dqn._select_action(observations[agent])
                else:
                    actions[agent] = 0  # good agent quieto

            next_obs, rewards, terminations, truncations, infos = self.env.step(actions)

            # --- colecciones para contraste de este paso ---
            z_contrast_inputs = {}   # guarda next_state_tensor por agente para recalcular z con grad
            optim_list = []          # embedding_optimizers a step() al final
            must_train = {}          # flags para cada agente (si toca entrenar en este paso)

            # ===== LOOP POR AGENTE: gating SIN grad + guardar replay + preparar contraste =====
            for agent in self.agent_dict:
                dqn = self.agent_dict[agent]
                done = terminations[agent] or truncations[agent]

                # 1) Embedding e y proyección z SIN GRAD para gating/registro
                next_state_np = np.array(next_obs[agent], dtype=np.float32)
                next_state_tensor = torch.tensor(next_state_np, dtype=torch.float32, device=dqn.device).unsqueeze(0)

                with torch.no_grad():
                    e_ng = dqn.embedding_net(next_state_tensor).squeeze(0)   # [64]
                    z_ng = dqn.proj_head(e_ng).squeeze(0)                    # [128], normalizado

                # 2) Novedad individual NGU + gating por novedad compartida
                intrinsic = dqn.memory.get_intrinsic_reward(e_ng.cpu().numpy())
                dqn.memory.add(e_ng.cpu().numpy())
                if not self.cosreg.is_team_novel(z_ng.cpu().numpy()):
                    intrinsic = 0.0
                self.cosreg.register(
                    agent_id=int(agent.split('_')[-1]) if agent.split('_')[-1].isdigit() else hash(agent) % 10**6,
                    z=z_ng.cpu().numpy()
                )

                # 3) Guardar experiencia
                dqn.replay_buffer.add(
                    np.array(observations[agent], dtype=np.float32),
                    actions[agent],
                    rewards[agent],
                    intrinsic,
                    next_state_np,
                    float(done)
                )
                episode_reward[agent] += rewards[agent]

                # 4) Guardar insumo para contraste (recalcularemos z CON grad, fuera del loop)
                z_contrast_inputs[agent] = next_state_tensor     # ya en device
                optim_list.append(dqn.embedding_optimizer)
                must_train[agent] = (t > dqn.learning_starts and t % dqn.train_freq == 0)

            # ===== 5) Pérdida contrastiva MULTI-AGENTE (con grad) ANTES de _train_step() =====
            z_tensors = []
            for agent in self.agent_dict:
                dqn = self.agent_dict[agent]
                x = z_contrast_inputs[agent]                             # [1, obs_dim]
                z = dqn.proj_head(dqn.embedding_net(x)).squeeze(0)      # [128], normalizado (con grad)
                z_tensors.append(z)

            if len(z_tensors) >= 2 and self.alpha_contrast > 0.0:
                # limpiar grads de todos los encoders+proyecciones
                for opt in optim_list:
                    opt.zero_grad()
                z_stack = torch.stack(z_tensors, dim=0)                  # [N, 128]
                L_contrast = contrastive_multiview_loss(z_stack, temp=self.temp_contrast)
                (self.alpha_contrast * L_contrast).backward()
                # aplicar step a cada optim (autograd enruta a los params correctos)
                for opt in optim_list:
                    opt.step()

            # ===== 6) AHORA sí: entrenamiento DQN / inverse dynamics =====
            for agent in self.agent_dict:
                dqn = self.agent_dict[agent]
                if must_train[agent]:
                    dqn._train_step()
                dqn._update_target_network()

            ep_len += 1
            t += 1
            observations = next_obs

            if terminations["agent_0"] or truncations["agent_0"]:
                self.ep_count += 1
                self.ep_lens.append(ep_len)
                self.episode_rewards_log.append(episode_reward.copy())
                for agent in self.agent_dict:
                    self.ep_rews[agent].append(episode_reward[agent])
                    episode_reward[agent] = 0
                    self.agent_dict[agent].memory.reset()
                ep_len = 0
                self.cosreg.reset()
                observations, _ = self.env.reset()

            if self.ep_count > 0 and self.ep_count % self.log_interval == 0 and self.ep_count != self.last_logged_ep:
                self._print_log(t)
                self.last_logged_ep = self.ep_count

    def share_replay_buffer(self, source_agent_name):
        shared_buffer = self.agent_dict[source_agent_name].replay_buffer
        for name in self.agent_dict:
            self.agent_dict[name].replay_buffer = shared_buffer

    def _print_log(self, t):
        fps = int(t / (time.perf_counter() - self.start_time))
        print("----------------------------------")
        print(f"| time/               |          |")
        print(f"|    episodes         | {self.ep_count}")
        print(f"|    fps              | {fps}")
        print(f"|    time_elapsed     | {int(time.perf_counter() - self.start_time)}")
        print(f"|    total_timesteps  | {t}")
        print(f"|    ep_len_mean      | {np.mean(self.ep_lens) if self.ep_lens else float('nan'):.0f}")
        for agent in self.agent_dict:
            dqn = self.agent_dict[agent]
            loss = dqn.last_loss if dqn.last_loss is not None else float("nan")
            print(f"| {agent} -> mean_ep_rew | {np.mean(self.ep_rews[agent]):.2f} | loss: {loss:.4f} | eps: {dqn.epsilon:.3f}")
        print("----------------------------------")

    def plot_total_rewards(self):
        """
        Plot de recompensas totales por episodio por agente.
        """
        if not self.episode_rewards_log:
            print("No hay recompensas registradas para graficar.")
            return

        agent_names = self.episode_rewards_log[0].keys()
        for agent in agent_names:
            rewards = [ep[agent].cpu().item() if torch.is_tensor(ep[agent]) else ep[agent] for ep in self.episode_rewards_log]
            plt.plot(rewards, label=agent)

        plt.xlabel("Episodio")
        plt.ylabel("Reward total por agente")
        plt.title("Recompensas por episodio")
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_rewards_to_csv(self, filename="rewards.csv"):
        cleaned_log = []

        for ep_idx, ep in enumerate(self.episode_rewards_log):
            row = {"episode": ep_idx}

            for agent, value in ep.items():
                if torch.is_tensor(value):
                    value = value.cpu().item()
                row[agent] = value

            cleaned_log.append(row)

        # Crear DataFrame
        df = pd.DataFrame(cleaned_log)
        df.to_csv(filename, index=False)

    def evaluate(self, episodes=5):
        rewards = []

        for agent in self.agent_dict.values():
            agent.epsilon = 0  # Evaluación sin exploración

        for _ in range(episodes):
            observations, _ = self.env.reset()
            total_rewards = {agent: 0 for agent in self.env.agents}
            terminated = {agent: False for agent in self.env.agents}
            truncated = {agent: False for agent in self.env.agents}

            while not all(terminated[agent] or truncated[agent] for agent in self.env.agents):
                actions = {}
                for agent in self.env.agents:
                    if terminated[agent] or truncated[agent]:
                        continue
                    if agent in self.agent_dict:
                        actions[agent] = self.agent_dict[agent]._select_action(observations[agent])
                    else:
                        actions[agent] = 0  # El agente bueno se queda quieto

                observations, rewards_step, terminated_step, truncated_step, _ = self.env.step(actions)

                for agent in self.env.agents:
                    total_rewards[agent] += rewards_step[agent]
                    terminated[agent] = terminated_step[agent]
                    truncated[agent] = truncated_step[agent]

            rewards.append(total_rewards)

        # Graficar
        agent_names = rewards[0].keys()
        for agent in agent_names:
            values = [ep[agent].cpu().item() if torch.is_tensor(ep[agent]) else ep[agent] for ep in rewards]
            plt.plot(values, label=agent)

        plt.xlabel("Episodio de evaluación")
        plt.ylabel("Reward total por agente")
        plt.title("Recompensas por episodio (evaluación)")
        plt.legend()
        plt.grid(True)
        plt.show()

        return rewards

    def render_and_save(self, num_tests=1, save_path="multiagent_episode.mp4", fps=5, max_steps=500):
        all_frames = []

        for agent in self.agent_dict.values():
            agent.epsilon = 0.0

        for _ in range(num_tests):
            observations, _ = self.env.reset()
            terminated = {agent: False for agent in self.env.agents}
            truncated = {agent: False for agent in self.env.agents}
            frames = []
            step_count = 0

            while not all(terminated[agent] or truncated[agent] for agent in self.env.agents) and step_count < max_steps:
                actions = {}
                for agent in self.env.agents:
                    if terminated[agent] or truncated[agent]:
                        continue
                    if agent in self.agent_dict:
                        actions[agent] = self.agent_dict[agent]._select_action(observations[agent])
                    else:
                        actions[agent] = 0

                observations, _, terminated_step, truncated_step, _ = self.env.step(actions)

                for agent in self.env.agents:
                    terminated[agent] = terminated_step[agent]
                    truncated[agent] = truncated_step[agent]

                frame = self.env.render()
                frame_resized = np.array(Image.fromarray(frame).resize((240, 240)))
                frames.append(frame_resized)
                step_count += 1

            all_frames.extend(frames + [frames[-1]] * 5)

        imageio.mimsave(save_path, all_frames, fps=fps)
        print(f"Video guardado en: {save_path}")