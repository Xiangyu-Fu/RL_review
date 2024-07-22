import gymnasium as gym

# import the skrl components to build the RL system
import torch
import torch.nn as nn
import numpy as np

from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.model_instantiators.torch import Shape, deterministic_model


# 创建环境
env = wrap_env(gym.make("CartPole-v1"))

device = env.device

# 创建Q网络实例
models = {}
models["q_network"] = deterministic_model(observation_space=env.observation_space,
                                          action_space=env.action_space,
                                          device=device,
                                          clip_actions=False,
                                          input_shape=Shape.OBSERVATIONS,
                                          hiddens=[64, 64],
                                          hidden_activation=["relu", "relu"],
                                          output_shape=Shape.ACTIONS,
                                          output_activation=None,
                                          output_scale=1.0)
models["target_q_network"] = deterministic_model(observation_space=env.observation_space,
                                                 action_space=env.action_space,
                                                 device=device,
                                                 clip_actions=False,
                                                 input_shape=Shape.OBSERVATIONS,
                                                 hiddens=[64, 64],
                                                 hidden_activation=["relu", "relu"],
                                                 output_shape=Shape.ACTIONS,
                                                 output_activation=None,
                                                 output_scale=1.0)

# initialize models' parameters (weights and biases)
for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# 创建经验回放缓冲区
memory = RandomMemory(memory_size=50000, num_envs=env.num_envs, device=device, replacement=False)

# 定义DQN智能体的超参数
cfg = DQN_DEFAULT_CONFIG.copy()
cfg["learning_starts"] = 100
cfg["exploration"]["final_epsilon"] = 0.04
cfg["exploration"]["timesteps"] = 1500
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 1000
cfg["experiment"]["checkpoint_interval"] = 5000
cfg["experiment"]["directory"] = "runs/torch/CartPole"

# 创建DQN智能体
agent = DQN(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

# 训练结束后，播放训练好的模型
import time

# 创建新的环境实例用于播放
play_env = wrap_env(gym.make("CartPole-v1", render_mode="human"))

# 载入训练好的模型参数
agent.load("runs/torch/CartPole/24-07-23_00-31-50-491014_DQN/checkpoints/best_agent.pt")

# 播放训练好的模型
num_episodes = 5
for episode in range(num_episodes):
    state = play_env.reset()
    done = False
    total_reward = 0
    timestep = 0

    while not done:
        # 利用训练好的模型选择动作
        action = agent.act(states=state[0], timestep=timestep, timesteps=50)
        play_env.step(action[0])
        play_env.render()  # 渲染环境
        time.sleep(0.02)  # 控制播放速度
        timestep += 1

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")


play_env.close()