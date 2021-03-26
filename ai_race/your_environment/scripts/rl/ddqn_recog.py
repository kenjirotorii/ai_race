#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

import random
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from networks.recognet import season_recog_net
from transition import Transition
from memory import Memory


class Brain:

    def __init__(self, 
                img_size=(240, 240),
                num_actions=3,
                num_z=256,
                mem_capacity=1000, 
                batch_size=32, 
                lr=1e-4, 
                gamma=0.99, 
                debug=True, 
                device='cpu',
                ae_model=None,
                recog_model=None):

        # parameters
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.debug = debug
        self.device = device

        # memory 
        self.memory = Memory(capacity=mem_capacity)

        # network
        self.main_q_network = season_recog_net(img_size, num_z, 4, num_actions, ae_model, recog_model)
        self.target_q_network = season_recog_net(img_size, num_z, 4, num_actions, ae_model, recog_model)

        self.main_q_network.to(self.device)
        self.target_q_network.to(self.device)

        # optimizer
        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=lr)

    def replay(self):
        '''Experience Replayでネットワークのパラメータを学習'''

        # 1. メモリサイズの確認
        if len(self.memory) < self.batch_size:
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
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, self.batch.next_state)), dtype=torch.bool).to(self.device)
        
        a_m = torch.zeros(self.batch_size).type(torch.LongTensor).to(self.device)

        # 次の状態での最大Q値の行動a_mをMain Q-Networkから求める
        # 最後の[1]で行動に対応したindexが返る
        a_m[non_final_mask] = self.main_q_network(
            self.non_final_next_states).detach().max(1)[1]

        # 次の状態があるものだけにフィルター
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        # まずは全部0にしておく
        next_state_values = torch.zeros(self.batch_size).to(self.device)

        # 次の状態があるindexの、最大となるQ値をtarget Q-Networkから求める
        next_state_values[non_final_mask] = self.target_q_network(self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

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

    def save_model(self, model_path):
        torch.save(self.main_q_network.state_dict(), model_path)

    def load_model(self, model_path):
        self.main_q_network.load_state_dict(torch.load(model_path))
        self.update_target_q_network()

    def decide_action(self, state, epsilon):
        '''現在の状態に応じて、行動を決定する'''
        # ε-greedy法で徐々に最適行動のみを採用する

        if epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval()  # ネットワークを推論モードに切り替える
            with torch.no_grad():
                output = self.main_q_network(state)
                if self.debug:
                    print("output={}".format(output.to('cpu')))
                # ネットワークの出力の最大値のindexを取り出す
                return output.max(1)[1].view(1, 1)

        else:
            # 0,1の行動をランダムに返す
            return  torch.LongTensor([[random.randrange(self.num_actions)]]).to(self.device)


class Agent:

    def __init__(self,
                img_size=(240, 240), 
                num_actions=3,
                num_z=256,
                mem_capacity=1000, 
                batch_size=32, 
                lr=1e-4, 
                gamma=0.99, 
                debug=True, 
                device='cpu',
                ae_model=None,
                recog_model=None):

        '''課題の状態と行動の数を設定する'''
        self.brain = Brain(img_size, num_actions, num_z, mem_capacity, batch_size, lr, 
                            gamma, debug, device, ae_model, recog_model)  # エージェントが行動を決定するための頭脳を生成

    def update_q_function(self):
        '''Q関数を更新する'''
        self.brain.replay()

    def get_action(self, state, epsilon):
        '''行動を決定する'''
        action = self.brain.decide_action(state, epsilon)
        return action

    def memorize(self, state, action, state_next, reward):
        '''memoryオブジェクトに、state, action, state_next, rewardの内容を保存する'''
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        '''Target Q-NetworkをMain Q-Networkと同じに更新'''
        self.brain.update_target_q_network()

    def save_model(self, model_path):
        self.brain.save_model(model_path)

    def load_model(self, model_path):
        self.brain.load_model(model_path)
