import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio
from PIL import Image
from collections import deque

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

            for agent in self.agent_dict:
                dqn = self.agent_dict[agent]
                done = terminations[agent] or truncations[agent]

                # Cálculo de embedding e intrínseco
                next_state_tensor = torch.tensor(next_obs[agent], dtype=torch.float32).unsqueeze(0).to(dqn.device)
                with torch.no_grad():
                    embed_tensor = dqn.embedding_net(next_state_tensor).squeeze(0).cpu().numpy()
                intrinsic = dqn.memory.get_intrinsic_reward(embed_tensor)
                dqn.memory.add(embed_tensor)

                dqn.replay_buffer.add(
                    np.array(observations[agent], dtype=np.float32),
                    actions[agent],
                    rewards[agent],
                    intrinsic,
                    np.array(next_obs[agent], dtype=np.float32),
                    float(done)
                )

                episode_reward[agent] += rewards[agent]

                if t > dqn.learning_starts and t % dqn.train_freq == 0:
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
                observations, _ = self.env.reset()

            if self.ep_count > 0 and self.ep_count % self.log_interval == 0 and self.ep_count != self.last_logged_ep:
                self._print_log(t)
                self.last_logged_ep = self.ep_count

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