#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
References:
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    https://github.com/YutaroOgawa/Deep-Reinforcement-Learning-Book/blob/master/program/6_3_DuelingNetwork.ipynb
'''

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

import random
import numpy as np
from collections import namedtuple

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from networks.mobilenet import MobileNetV2


Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward')
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Memory:

    def __init__(self, capacity):
        self.capacity = capacity  # メモリの最大長さ
        self.memory = []  # 経験を保存する変数
        self.index = 0  # 保存するindexを示す変数

    def push(self, state, action, state_next, reward):
        '''transition = (state, action, state_next, reward)をメモリに保存する'''

        if len(self.memory) < self.capacity:
            self.memory.append(None)  # メモリが満タンでないときは足す

        # namedtupleのTransitionを使用し、値とフィールド名をペアにして保存します
        self.memory[self.index] = Transition(state, action, state_next, reward)

        self.index = (self.index + 1) % self.capacity  # 保存するindexを1つずらす

    def sample(self, batch_size):
        '''batch_size分だけ、ランダムに保存内容を取り出す'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        '''関数lenに対して、現在の変数memoryの長さを返す'''
        return len(self.memory)


class Brain:

    def __init__(self, num_actions, mem_capacity, batch_size, lr, gamma):

        # parameters
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma

        # memory 
        self.memory = Memory(capacity=mem_capacity)

        # network
        self.main_q_network = MobileNetV2(num_classes=num_actions).to(DEVICE)
        self.target_q_network = MobileNetV2(num_classes=num_actions).to(DEVICE)

        #print(self.main_q_network)

        # optimizer
        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=lr)

    def replay(self):
        '''Experience Replayでネットワークのパラメータを学習'''

        # 1. メモリサイズの確認
        if len(self.memory) < self.batch_size:
            print("Skip training because memory size is less than batch size")
            return

        # 2. ミニバッチの作成
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()

        # 3. 教師信号となるQ(s_t, a_t)値を求める
        self.expected_state_action_values = self.get_expected_state_action_values()

        # 4. 結合パラメータの更新
        self.update_main_q_network()

    def make_minibatch(self):
        '''2. ミニバッチの作成'''

        # 2.1 メモリからミニバッチ分のデータを取り出す
        transitions = self.memory.sample(self.batch_size)

        # 2.2 各変数をミニバッチに対応する形に変形
        batch = Transition(*zip(*transitions)) # Transition(state=(tensor(...), tensor(...), ...), action=(...), ...)

        # 2.3 各変数の要素をミニバッチに対応する形に変形し、ネットワークで扱えるようVariableにする
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action) # tensor([[1], [2], ....]), size=torch.Size([batch_size, 1])
        reward_batch = torch.cat(batch.reward) # tensor([[1.0], [0.0], ....]), size=torch.Size([batch_size, 1])

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        '''3. 教師信号となるQ(s_t, a_t)値を求める'''

        # 3.1 ネットワークを推論モードに切り替える
        self.main_q_network.eval()
        self.target_q_network.eval()

        # 3.2 ネットワークが出力したQ(s_t, a_t)を求める
        self.state_action_values = self.main_q_network(self.state_batch).gather(1, self.action_batch)

        # 3.3 max{Q(s_t+1, a)}値を求める。ただし次の状態があるかに注意

        # next_stateがあるかをチェックするインデックスマスクを作成
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, self.batch.next_state)), dtype=torch.bool).to(DEVICE)

        # まずは全部0にしておく
        next_state_values = torch.zeros(self.batch_size).to(DEVICE)

        # 次の状態があるindexの、最大となるQ値をtarget Q-Networkから求める
        next_state_values[non_final_mask] = self.target_q_network(self.non_final_next_states).max(1)[0].detach()

        # 3.4 教師となるQ(s_t, a_t)値を、Q学習の式から求める
        expected_state_action_values = self.reward_batch + self.gamma * next_state_values

        return expected_state_action_values

    def update_main_q_network(self):
        '''4. networkパラメータの更新'''

        # 4.1 ネットワークを訓練モードに切り替える
        self.main_q_network.train()

        # 4.2 損失関数を計算する（smooth_l1_lossはHuberloss）
        loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1))

        # 4.3 結合パラメータを更新する
        self.optimizer.zero_grad()  # 勾配をリセット
        loss.backward()  # バックプロパゲーションを計算
        self.optimizer.step()  # 結合パラメータを更新

    def update_target_q_network(self):
        '''Target Q-NetworkをMainと同じにする'''
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

    def decide_action(self, state, episode):
        '''現在の状態に応じて、行動を決定する'''
        # ε-greedy法で徐々に最適行動のみを採用する
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval()  # ネットワークを推論モードに切り替える
            with torch.no_grad():
                # ネットワークの出力の最大値のindexを取り出す
                return self.main_q_network(state).max(1)[1].view(1, 1)

        else:
            # 0,1の行動をランダムに返す
            return  torch.LongTensor([[random.randrange(self.num_actions)]]).to(DEVICE)


class Agent:

    def __init__(self, num_actions=3, mem_capacity=1000, batch_size=32, lr=0.0001, gamma=0.99):
        '''課題の状態と行動の数を設定する'''
        self.brain = Brain(num_actions, mem_capacity, batch_size, lr, gamma)  # エージェントが行動を決定するための頭脳を生成

    def update_q_function(self):
        '''Q関数を更新する'''
        self.brain.replay()

    def get_action(self, state, episode):
        '''行動を決定する'''
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        '''memoryオブジェクトに、state, action, state_next, rewardの内容を保存する'''
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        '''Target Q-NetworkをMain Q-Networkと同じに更新'''
        self.brain.update_target_q_network()
