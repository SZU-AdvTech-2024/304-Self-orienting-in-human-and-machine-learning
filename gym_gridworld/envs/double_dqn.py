import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import json
import time
from tqdm import tqdm  # 添加进度条显示

# 设置环境变量和路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 导入本地模块
from gridworld_env import GridworldEnv
from params.param_dicts import params

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, shape[2], shape[0], shape[1], device=next(self.parameters()).device))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # 添加批次维度
        x = x.permute(0, 3, 1, 2)  # (batch, height, width, channels) -> (batch, channels, height, width)
        x = x.contiguous()  # 确保内存连续
        conv_out = self.conv(x)
        conv_out = conv_out.flatten(1)  # 使用 flatten 代替 view
        return self.fc(conv_out)

class DoubleDQNAgent:
    def __init__(self, env, learning_rate=1e-4, gamma=0.99, epsilon_start=1.0,
                 epsilon_final=0.01, epsilon_decay=10000, memory_size=100000,
                 batch_size=32, target_update=1000, device="cuda" if torch.cuda.is_available() else "cpu"):
        
        self.env = env
        self.n_actions = env.action_space.n
        self.device = device
        
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.memory = deque(maxlen=memory_size)
        self.target_update = target_update
        
        self.policy_net = DQN(env.observation_space.shape, self.n_actions).to(device)
        self.target_net = DQN(env.observation_space.shape, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        self.metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "losses": []
        }
        
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            state = state.contiguous()
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def update_epsilon(self, frame_idx):
        self.epsilon = max(self.epsilon_final, 
                         self.epsilon - (1 - self.epsilon_final) / self.epsilon_decay)
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        state_batch = torch.FloatTensor(np.array([s[0] for s in batch])).to(self.device)
        action_batch = torch.LongTensor(np.array([s[1] for s in batch])).to(self.device)
        reward_batch = torch.FloatTensor(np.array([s[2] for s in batch])).to(self.device)
        next_state_batch = torch.FloatTensor(np.array([s[3] for s in batch])).to(self.device)
        done_batch = torch.FloatTensor(np.array([s[4] for s in batch])).to(self.device)
        
        # 确保张量内存连续
        state_batch = state_batch.contiguous()
        next_state_batch = next_state_batch.contiguous()
        
        q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        with torch.no_grad():
            policy_next_q_values = self.policy_net(next_state_batch)
            policy_next_actions = policy_next_q_values.max(1)[1].unsqueeze(1)
            
            next_q_values = self.target_net(next_state_batch)
            next_q_values = next_q_values.gather(1, policy_next_actions)
            
            expected_q_values = reward_batch.unsqueeze(1) + (1 - done_batch.unsqueeze(1)) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_episodes, save_dir="checkpoints"):
        os.makedirs(save_dir, exist_ok=True)
        frame_idx = 0
        best_reward = float('-inf')
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                episode_reward += reward
                episode_length += 1
                frame_idx += 1
                
                loss = self.optimize_model()
                if loss is not None:
                    self.metrics["losses"].append(loss)
                
                if frame_idx % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                
                self.update_epsilon(frame_idx)
                
                if done:
                    break
            
            self.metrics["episode_rewards"].append(episode_reward)
            self.metrics["episode_lengths"].append(episode_length)
            
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save({
                    'episode': episode,
                    'model_state_dict': self.policy_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'reward': episode_reward,
                }, f"{save_dir}/best_model.pt")
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.metrics["episode_rewards"][-100:])
                avg_length = np.mean(self.metrics["episode_lengths"][-100:])
                print(f"Episode {episode + 1}")
                print(f"Average Reward (last 100): {avg_reward:.2f}")
                print(f"Average Length (last 100): {avg_length:.2f}")
                print(f"Epsilon: {self.epsilon:.3f}")
                print("----------------------------------------")
        
        return self.metrics

def train_and_save_levels(num_sets=1, levels_per_set=500):
    base_params = {
        'game_type': 'logic',
        'player': 'dqn_training',
        'exp_name': 'ddqn_experiment',
        'singleAgent': True,
        'verbose': False,
        'log_neptune': False,
        'n_levels': levels_per_set,
        'single_loc': False,
        'shuffle_keys': False,
        'shuffle_each': 1,
        'agent_location_random': True,
        'different_self_color': False,
        'switch_self_finding_100_lvls': False,
        'ten_r': False,
        'save': True,
        'mid_modify': False,
        'neg_reward': True,
        'levels_count': 5,
        'seed': 42  # 添加seed参数
    }

    for set_idx in range(num_sets):
        print(f"\nTraining Set {set_idx + 1}")
        
        # 为每个set创建独立的存储目录
        set_save_path = f'./models/set_{set_idx + 1}/'
        set_data_dir = f'./levels/set_{set_idx + 1}/'
        os.makedirs(set_save_path + 'lastSave', exist_ok=True)
        os.makedirs(set_data_dir, exist_ok=True)
        
        # 更新当前set的存储路径和seed
        current_params = base_params.copy()
        current_params.update({
            'save_path': set_save_path,
            'data_save_dir': set_data_dir,
            'seed': base_params['seed'] + set_idx  # 每个set使用不同的seed
        })
        
        env = GridworldEnv()
        env.make_game(current_params)
        
        agent = DoubleDQNAgent(env)
        agent.train(num_episodes=300)
        
        print(f"Completed Set {set_idx + 1}")
        env.reset()
       

def main():
    try:
        # 只需创建基础目录
        os.makedirs("./levels", exist_ok=True)
        os.makedirs("./models", exist_ok=True)
        os.makedirs("./checkpoints", exist_ok=True)
        
        train_and_save_levels(num_sets=20, levels_per_set=300)
    except Exception as e:
        print(f"Error occurred: {e}")
        raise e

if __name__ == "__main__":
    main()