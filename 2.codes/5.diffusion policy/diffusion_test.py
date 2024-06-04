import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DenoiseModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DenoiseModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state, action, t):
        x = torch.cat([state, action, t], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# add noise to action
def forward_diffusion(action, alpha):
    return np.sqrt(alpha) * action + np.sqrt(1 - alpha) * np.random.normal(size=action.shape)

def train_denoise_model(model, optimizer, states, actions, rewards, num_steps=1000):
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # t 为随机生成的参数，用于控制噪声的大小
        t = np.random.uniform(0, 1, size=(states.shape[0], 1))
        alpha = 1 - t  # 简单示例，实际应根据具体过程定义
        
        noisy_actions = forward_diffusion(actions, alpha)  # 添加噪声

        # 转换为 PyTorch 的 Tensor
        noisy_actions = torch.tensor(noisy_actions, dtype=torch.float32)
        states_torch = torch.tensor(states, dtype=torch.float32)
        t_torch = torch.tensor(t, dtype=torch.float32)
        
        # 使用去噪模型预测动作
        pred_actions = model(states_torch, noisy_actions, t_torch)  
        
        # 使用强化学习的奖励信号来优化去噪模型
        loss = nn.MSELoss()(pred_actions, torch.tensor(actions, dtype=torch.float32)) - torch.mean(torch.tensor(rewards, dtype=torch.float32))  # 优化目标
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item()}")

# 示例数据
states = np.random.rand(100, 4)  # 100 个状态，每个状态 4 维
actions = np.random.rand(100, 2)  # 100 个动作，每个动作 2 维
rewards = np.random.rand(100, 1)  # 100 个奖励，每个奖励 1 维

# 初始化模型和优化器
model = DenoiseModel(state_dim=4, action_dim=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
train_denoise_model(model, optimizer, states, actions, rewards)
