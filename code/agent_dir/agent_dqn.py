import os
import random
import copy
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tensorboardX import SummaryWriter
from torch import nn, optim
from agent_dir.agent import Agent
from collections import deque
import math
class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))  # 使用ReLU激活函数作用在第一个全连接层的输出上
        x = self.fc2(x)  # 第二个全连接层，输出即为最终的Q值
        return x




class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def push(self, *transition):
        """添加经验，以五元元组方式存储，假如样本池已满则清除一个样本。"""
        self.buffer.append(transition)
        if len(self.buffer)==self.buffer_size:
            self.clean()

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch

    def clean(self):        
        self.buffer.clear()



class AgentDQN(Agent):
    def __init__(self, env, args):
        super(AgentDQN, self).__init__(env)
        self.args=args
        self.q_network=QNetwork(4,args.hidden_size,2)
        self.target_network=QNetwork(4,args.hidden_size,2)# 经验池

        self.buffer=ReplayBuffer(args.buffer_size)
        # 用于tensorboard画图
        self.writer=SummaryWriter("epsilon=0.95")
        # 计数
        self.counter=0

        # 设置随机数种子
        np.random.seed(self.args.seed)

         # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.lr)
        # 损失函数
        self.loss_fn=nn.MSELoss()
    
    def init_game_setting(self):
        pass

    def train(self):
        if len(self.buffer) < self.args.batch_size: 
            return
        transitions = self.buffer.sample(self.args.batch_size)
        batch = list(zip(*transitions))
        state_batch = torch.tensor(batch[0], dtype=torch.float32)
        action_batch = torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1)
        next_state_batch = torch.tensor(batch[3], dtype=torch.float32)
        done_batch = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1)
        
        current_q_values = self.q_network(state_batch).gather(1, action_batch)
        
        # 获得下一状态的最大q值不需要计算目标网络的梯度
        next_q_values = self.target_network(next_state_batch).max(1)[0].detach().unsqueeze(1)
        target_q_values = reward_batch + (1 - done_batch) * self.args.gamma * next_q_values

        loss = self.loss_fn(target_q_values, current_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.writer.add_scalar("loss",loss.item(),self.counter)
        self.counter += 1
        if self.counter % self.args.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    # def train(self):
    #     """
    #     Implement your training algorithm here
    #     """
    #     # 取样
    #     transitions = self.buffer.sample(self.args.batch_size)
    #     s,a,r,s_,flag=list(zip(*transitions))
    #     # 将相应数据转化成tensor形式，方便使用pytorch
    #     s=torch.from_numpy(np.array(s))
    #     s_=torch.from_numpy(np.array(s_))
    #     r=torch.tensor([[i] for i in r],dtype=torch.float32)
    #     a=torch.tensor([[i] for i in a],dtype=torch.int64)
    #     # 获得下一状态的最大q值不需要计算目标网络的梯度
    #     with torch.no_grad():
    #     # 计算最大下一状态的q值
    #         max_q_next=torch.max(self.target_network(s_),dim=1)[0]
    #     # 根据样本中观察和动作估计q值
    #     q_value=self.q_network(s).gather(1,a)
    #     # 计算TDerror
    #     q_target=[]
    #     for i in range(len(max_q_next)):
    #         # 如果flag为真，说明下一状态不存在，不需要加入其q值估计
    #         if flag[i]:
    #             q_target.append([r[i]])
    #         # 反之需要加入q值估计
    #         else:
    #             q_target.append([r[i]+self.args.gamma*max_q_next[i]])
    #     # TDerror
    #     q_target=torch.tensor(q_target)
    #     # 计算loss，这里采用MESloss
    #     loss=self.loss_fn(q_target,q_value)
        
    #     # 梯度清空
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     self.writer.add_scalar("loss",loss.item(),self.counter)
    #     # 计数增加
    #     self.counter+=1
    #     # 每d次更新目标网络
    #     if self.counter%self.args.target_update==0:
    #         self.target_network.load_state_dict(self.q_network.state_dict())

    def make_action(self, observation, test=True):
        ob = torch.tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            # 估计q值
            qa=self.q_network(ob)
            # q值最大的动作
            greedy_action=qa.argmax().item()
        # ε-greedy策略
        if np.random.random()<self.args.epsilon:
            # 以ε的概率采取最优策略
            return greedy_action
        else:
            # 不然选取非最优策略
            return 1-greedy_action
        
    

        
    def run(self):
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        t = 0
        for episode in range(self.args.episodes):
            self.init_game_setting()
            state = self.env.reset()
            episode_reward = 0

            while True:
                action = self.make_action(observation=state, test=False)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                self.buffer.push(state, action, reward, next_state, done)
                state = next_state
                t+=1

                if t >= self.args.batch_size+1:
                    self.train()
                    #t-=1

                if done:
                    break

            print(f"Episode {episode}: Reward = {episode_reward}")
            # 将每个episode的奖励写入Tensorboard
            self.writer.add_scalar('Reward/Episode', episode_reward, episode)

        # 关闭TensorboardX SummaryWriter
        self.writer.close()