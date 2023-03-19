from distutils.log import info
from tabnanny import check
import numpy as np
import os 
import torch
from torch import nn
from torch.distributions.normal import Normal
from datetime import datetime, time
import matplotlib.pyplot as plt
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

'''
env_name = './Racer/Racer.exe'
unity_env = UnityEnvironment(file_name=env_name)
env = UnityToGymWrapper(unity_env=unity_env, allow_multiple_obs=False)

env.reset()

env.close()
'''

# trajectory buffer
class Buffer():

    def __init__(self, batch_size):
        self.clear()
        self.batch_size = batch_size
  
    def record(self, obs, act, rew, don, val, logp):
        rew = torch.as_tensor(rew)
        don = torch.as_tensor(don)

        self.obs = torch.cat((self.obs,obs.unsqueeze(0)),dim=0)
        self.act = torch.cat((self.act,act.unsqueeze(0)),dim=0)
        self.rew = torch.cat((self.rew,rew.unsqueeze(0)),dim=0)
        self.don = torch.cat((self.don,don.unsqueeze(0)),dim=0)
        self.val = torch.cat((self.val,val),dim=0)
        self.logp = torch.cat((self.logp,logp),dim=0)

    def sample_batch(self):
        size = len(self.obs)
        batch_range = np.arange(0, size, self.batch_size)
        indices = np.arange(size, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_range]

        assert len(self.obs.shape) == 2 and self.obs.shape[1] == 5, f'obs shape is {self.obs.shape} but not (-1,5)'
        assert len(self.act.shape) == 2 and self.act.shape[1] == 2, f'obs shape is {self.act.shape} but not (-1,2)'
        assert len(self.rew.shape) == 1 , f'rew shape is {self.rew.shape} but not (-1,)'
        assert len(self.don.shape) == 1 , f'rew shape is {self.don.shape} but not (-1,)'
        assert len(self.val.shape) == 1 , f'rew shape is {self.val.shape} but not (-1,)'
        assert len(self.logp.shape) == 1 , f'rew shape is {self.logp.shape} but not (-1,)'

        return self.obs, self.act, self.rew,self.don, self.val, self.logp, batches
    def clear(self):
        self.obs = torch.empty((0,5), dtype=torch.float64)
        self.act = torch.empty((0,2), dtype=torch.float64)
        self.rew = torch.empty((0,), dtype=torch.float64)
        self.don = torch.empty((0,), dtype=torch.bool)
        self.val = torch.empty((0,), dtype=torch.float64)
        self.logp = torch.empty((0,), dtype=torch.float64)


# 狀態空間
NUM_STATES = 5
# 動作空間
NUM_ACTIONS = 2
# 動作上限
ACT_UPPER_BOUND = 1
# 動作下限
ACT_LOWER_BOUND = -1
# Mini-batch size for training
BATCH_SIZE = 64
# 模型權重
WHIGHTS_FILE = './weights/'
WEIGHTS_FILE_ACTOR = './weights/ppo_actor_model.pt'
WEIGHTS_FILE_CRITIC = './weights/ppo_critic_model.pt'

if not os.path.exists(WHIGHTS_FILE):
    os.mkdir(WHIGHTS_FILE)

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(NUM_STATES, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, NUM_ACTIONS),
            nn.Tanh()
        )
        self.sigma = 0.2
    def forward(self, obs):
        '''
        mu shape : (2, )
        '''
        mu = self.fc_layer(obs)
        return mu, self.sigma
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(NUM_STATES, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
    def forward(self, obs):
        '''
        output shape : (1, )
        '''
        x = self.fc_layer(obs)
        return x


class Logger():
    def __init__(self):
        self.epoch_logs = dict()
        self.iter_logs = dict()
        self.statistic = dict()
    def store_epoch(self, key, value):
        if key not in self.epoch_logs.keys():
            self.epoch_logs[key] = []
        self.epoch_logs[key].append(value)
    def store_iter(self, key, value):
        if key not in self.iter_logs.keys():
            self.iter_logs[key] = []
        self.iter_logs[key].append(value)
    def log_tabular(self, key, stats):
        # sum, min, max, mean
        self.statistic[key] = stats
    def dump_tabular(self):
        for k in self.iter_logs.keys():
            if k in self.statistic.keys() and self.statistic[k] == 'sum':
                self.store_epoch(f'{k}',np.sum(self.iter_logs[k]))
            else :
                self.store_epoch(f'{k}',np.mean(self.iter_logs[k]))
        self.iter_logs = dict()
    def render(self):
        key_lens = [len(key) for key in self.epoch_logs]
        max_key_len = max(15,max(key_lens))
        keystr = '%'+'%d'%max_key_len
        fmt = "| " + keystr + "s | %15s |"
        print('-'*37)
        for k in self.epoch_logs:
            valstr = "%8.3g"%self.epoch_logs[k][-1] if hasattr(self.epoch_logs[k][-1], "__float__") else self.epoch_logs[k][-1]
            print(fmt%(k, valstr))
        print('-'*37)

class Agent():
    def __init__(self, epochs, train_pi_v_pre_epoch=10, gamma=0.99, gae_lambda=0.95,
                clip_ratio = 0.25, target_entropy=0.01, pi_lr = 3e-4, v_lr = 3e-4, 
                target_kl=0.01):
        '''
        Args:
            epochs (int): 迭代次數
            iters_pre_epoch (int) : 每個 epoch 跟環境互動的次數
            gamma (float)
            gae_lambda (float)
            train_pi_v_pre_epoch (int) : 在相同的 epoch 更新 價值網路 和動作價值網路的次數
            clip_ratio (float) : 超參數，在policy網路時，需要用來裁減超過的部分
            target_entropy (float) : 超參數，在policy網路時用到(spinup 的ppo程式 沒有此部分需要研究)
            pi_lr (float) : policy 網路的學習率 
            v_lr (float) : value 網路的學習率
            self.target_KL (float) : 用來評估當前的policy跟舊的policy差距有多大，超過最大差距就要停止更新重新收集資料，常用(0.01, 0.05)
        '''

        self.epochs = epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.train_pi_v_pre_epoch = train_pi_v_pre_epoch
        self.clip_ratio = clip_ratio
        self.target_entropy = target_entropy
        self.target_kl = target_kl
        # 環境
        '''
        env_name = './Racer/Racer.exe'
        channel = EngineConfigurationChannel()
        unity_env = UnityEnvironment(file_name=env_name, side_channels=[channel])
        # channel.set_configuration_parameters(width=100, height=100, time_scale=20.0)
        channel.set_configuration_parameters(time_scale=20.0)
        
        self.racer = UnityToGymWrapper(unity_env=unity_env, allow_multiple_obs=False)
        '''
        # policy 網路
        self.actor = Actor()
        # 價值網路
        self.critic = Critic()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=pi_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=v_lr)

        # 經驗回放
        self.buffer = Buffer(BATCH_SIZE)
        # 紀錄
        self.logger = Logger()
    def gae(self, val, rew, done, last_value=0):

        ret = torch.empty((0,), dtype=torch.float64)
        gae = 0
        for i in reversed(range(len(rew))):
      
            if i == len(rew)-1:
                nextval = last_value
            else:
                nextval = val[i+1]

            delta = rew[i] + self.gamma * nextval * done[i] - val[i]
            gae = delta + self.gamma * self.gae_lambda * done[i] * gae
      
            ret = torch.cat(((gae+val[i]).unsqueeze(0), ret), dim=0)
        adv = torch.as_tensor(ret-val)
        # torch 中的 std 有 貝塞爾校正（Bessel's Correction）所以 unbiased 要把此校正關閉
        adv = (adv - torch.mean(adv)) / (torch.std(adv, unbiased=False)+1e-8)
        return ret, adv

    # def update_network(self, obs, act, rew, done, val, old_logp, batches, last_value=0):
    def update_network(self, last_value=0):
        obs, act, rew, done, val, old_logp, batches = self.buffer.sample_batch()
        ret, adv = self.gae(val, rew, done, last_value)
    
        #使用 mini-batches 訓練
        for batch in batches:

            obs_b = torch.as_tensor(obs[batch], dtype=torch.float32)
            act_b = torch.as_tensor(act[batch])
            adv_b = torch.as_tensor(adv[batch])
            ret_b = torch.as_tensor(ret[batch])
            ologp_b = torch.as_tensor(old_logp[batch])
      
            for iter in range(self.train_pi_v_pre_epoch):
        
                # 計算 loss pi
                _, logp_b = self.get_act_and_logp(obs_b, act_b)
        
                ratio = torch.exp(logp_b-ologp_b)
                clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv_b
                ## ***** -self.target_entropy * logp_b 需要研究 *****
                min_weighted_ratio = torch.min(ratio*adv_b, clip_adv) - self.target_entropy * logp_b
                loss_pi = - torch.mean(min_weighted_ratio)
                # 更新loss pi
                self.actor_optimizer.zero_grad()
                loss_pi.backward()
                self.actor_optimizer.step()

                # 計算 loss value (MSE)
                criterion = nn.MSELoss()
                loss_v = criterion(self.critic(obs_b).to(torch.float64).squeeze(),ret_b)

                # 更新 loss v
                self.critic_optimizer.zero_grad()
                loss_v.backward()
                self.critic_optimizer.step()

                # 為了早點停止更新，粗率計算 KL divergence
                with torch.no_grad():
                    _, logp = self.get_act_and_logp(obs_b, act_b)
          
                    kl = (ologp_b-logp).mean()
                    if kl > 1.5 * self.target_kl:
                        print("early stopping -  kl reached at iter {}".format(iter))
                        break
        self.logger.store_epoch('KL', kl.item())
        self.logger.store_epoch('LossPi', loss_pi.item())
        self.logger.store_epoch('LossV', loss_v.item())
        # 更新完網路清空資料重新收集數據
        self.buffer.clear()


    def get_act_and_logp(self, obs, act=None):
        #從 actor 拿到平均和標準差
        mu, sigma = self.actor(obs)

        #計算常態分布
        nor_pdf = Normal(mu, sigma)

        if act is None:
            # 動作不存在就sample一個動作
            act = mu + sigma * Normal(0, 1).sample((1,NUM_ACTIONS))
        # 取得當前動作的機率
        logp = nor_pdf.log_prob(act)

        if len(logp.shape) > 1:
            logp = torch.sum(logp, 1)
        else:
            logp = torch.sum(logp)
    
        #對動作進行驗證, 檢查動作上下限
        valid_act = torch.clamp(act, ACT_LOWER_BOUND, ACT_UPPER_BOUND)
        return valid_act, logp
    def step(self, action):
        '''
        有一定的機會agnet 什麼都不做就只往前走
        '''
        t = np.random.randint(0, 2)
        state, reward, done, info = self.racer.step(action)
        for i in range(t):
            if not done:
                state, t_r, done, info = self.racer.step([0, 0])
                reward += t_r
        return (state, reward, done)
    def train(self):
        # 平均速度
        mean_speed = 0
        max_reward = -3
        for epoch in range(self.epochs):
            self.logger.store_epoch('Epoch', epoch)
            obs = self.racer.reset()
      
            # 只玩一場遊戲
            done = False
            while not done:
                obs = torch.as_tensor(obs, dtype=torch.float32)
                # 取得 動作 和 此動作的機率
                with torch.no_grad():
                    act, logp = self.get_act_and_logp(obs)
        
                # 當前狀態價值
                val = self.critic(obs).clone().detach()
                # 執行動作後的 狀態, 獎勵, 是否結束
                act = act.squeeze()
                next_obs, rew, done = self.step(act)

                # 儲存經驗
                self.buffer.record(obs, act, rew, not done, val, logp)
        
                # 如果遊戲未結束
                if not done:
                    #儲存速度
                    self.logger.store_iter('Speed',next_obs[4])
                obs = next_obs
                # 紀錄 reward
                self.logger.store_iter('Rew', rew)

            # 遊戲結束就更新網路
            self.update_network()
            self.logger.log_tabular('Rew', 'sum')
            self.logger.dump_tabular()
            self.logger.render()
            if self.is_training and max_reward < self.logger.epoch_logs['Rew'][-1]:
                max_reward = self.logger.epoch_logs['Rew'][-1]
                print(f'save {max_reward}')
                torch.save(self.actor, WEIGHTS_FILE_ACTOR)
                torch.save(self.critic, WEIGHTS_FILE_CRITIC)
        if self.epochs > 0:
            if self.is_training:
                max_reward = self.logger.epoch_logs['Rew'][-1]
                print(f'save {max_reward}')
                torch.save(self.actor, WHIGHTS_FILE+"last_Actor_weight.pt")
                torch.save(self.critic, WHIGHTS_FILE+"last_Critic_weight.pt")
            plt.plot(np.convolve(self.logger.epoch_logs['Rew'], np.ones(40), 'valid') / 40)
            plt.xlabel("Episode")
            plt.ylabel("Avg. Episodic Reward")
            #plt.ylim(-3.5,7)
            plt.show(block=False)
            plt.savefig(f'./{datetime.now().strftime("%Y%m%d")}.png')
            plt.pause(0.001)
        self.racer.close()
    def eval(self):
        for i in range(2):
            obs = self.racer.reset()
            # 只玩一場遊戲
            done = False
            while not done:
                obs = torch.as_tensor(obs, dtype=torch.float32)
                # 取得 動作 和 此動作的機率
                with torch.no_grad():
                    act, logp = self.get_act_and_logp(obs)
            
                    # 當前狀態價值
                val = self.critic(obs).clone().detach()
                # 執行動作後的 狀態, 獎勵, 是否結束
                act = act.squeeze()

                next_obs, rew, done = self.step(act)
                obs = next_obs
        self.racer.close()

    def launch(self, is_training):
        self.is_training = is_training

        env_name = './Racer/Racer.exe'
        

        if is_training:

            channel = EngineConfigurationChannel()
            unity_env = UnityEnvironment(file_name=env_name, side_channels=[channel])
            channel.set_configuration_parameters(width=100, height=100, time_scale=20.0)
            self.racer = UnityToGymWrapper(unity_env=unity_env, allow_multiple_obs=False)

            start_t = datetime.now()
            self.train()
            end_t = datetime.now()
            print(f"Time elapsed : {end_t-start_t}")
        else:
            self.actor = torch.load(WEIGHTS_FILE_ACTOR)
            self.actor.eval()
            self.critic = torch.load(WEIGHTS_FILE_CRITIC)
            self.critic.eval()

            channel = EngineConfigurationChannel()
            unity_env = UnityEnvironment(file_name=env_name, side_channels=[channel])
            channel.set_configuration_parameters(width=1500, height=1000,time_scale=0)
            self.racer = UnityToGymWrapper(unity_env=unity_env, allow_multiple_obs=False)
            self.eval()
ppo = Agent(epochs=500, train_pi_v_pre_epoch=10)
ppo.launch(False)


