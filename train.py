import gym
import time
import numpy as np
import ppo
import torch
import os
import cv2
import matplotlib.pylab as plt
import env_wrapper
import rollout as roll
import argparse
import agent as agent
import time
print(torch.__version__)
from torch.utils.tensorboard import SummaryWriter

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.cuda.set_device(0)

writer = SummaryWriter(log_dir='./logs/')

try:  
    os.mkdir('./saved_actor')
except OSError:  
    print ("Creation of the directory failed")
else:  
    print ("Successfully created the directory")

"""
NUM_ACTION = 4
ENV_GAME_NAME = 'Breakout-v0'
VALUE_FACTOR = 1.0
ENTROPY_FACTOR = 0.01
EPSILON = 0.1
LR = 2.5e-4
LR_DECAY = 'linear'
GRAD_CLIP = 0.5#0.5
TIME_HORIZON = 128
BATCH_SIZE = 32 #32
GAMMA = 0.99
LAM = 0.95
EPOCH = 4
ACTORS = 8
FINAL_STEP = 10e6
    parser.add_argument('--value-factor', type=float, default=1.0)
    parser.add_argument('--entropy-factor', type=float, default=0.0)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--learning-rate-decay', type=str, default='constant')
    parser.add_argument('--grad-clip', type=float, default=0.5)
    parser.add_argument('--saved-episode', type=int, default=50)

    parser.add_argument('--time-horizon', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=64) #64
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)

    parser.add_argument('--epoch', type=int, default=10) #10
    parser.add_argument('--actors', type=int, default=1)
    parser.add_argument('--final-step', type=int, default=10e6)

    parser.add_argument('--value-factor', type=float, default=1.0)
    parser.add_argument('--entropy-factor', type=float, default=0.01)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--learning-rate', type=float, default=2.5e-4)
    parser.add_argument('--learning-rate-decay', type=str, default='constant')
    parser.add_argument('--grad-clip', type=float, default=0.5)
    parser.add_argument('--saved-episode', type=int, default=50)

    parser.add_argument('--time-horizon', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=32) #64
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)

    parser.add_argument('--epoch', type=int, default=4) #10
    parser.add_argument('--actors', type=int, default=8)
    parser.add_argument('--final-step', type=int, default=10e6)
"""

def parse_args():
    parser = argparse.ArgumentParser('Reinforcement Learning parser for PPO')

    parser.add_argument('--scenario', type=str, default='LunarLander-v2') #LunarLander-v2
    parser.add_argument('--eval', action='store_false') #LunarLander-v2

    #parser.add_argument('--num-action', type=int, default=3)
    parser.add_argument('--value-factor', type=float, default=1.0)
    parser.add_argument('--entropy-factor', type=float, default=0.01)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--learning-rate', type=float, default=2.5e-4)
    parser.add_argument('--learning-rate-decay', type=str, default='linear')
    parser.add_argument('--grad-clip', type=float, default=0.5)

    parser.add_argument('--load-episode-saved', type=int, default=50)
    parser.add_argument('--saved-episode', type=int, default=500)

    parser.add_argument('--time-horizon', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=32) #32
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)

    parser.add_argument('--epoch', type=int, default=3) #10
    parser.add_argument('--actors', type=int, default=8)
    parser.add_argument('--final-step', type=int, default=10e6) #10e6
    return parser.parse_args()

def _process_obs(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    return obs[None, None, :, :] / 255 # Shape (84, 84, 1)
    #return np.reshape(obs, (1, 6, 1))

def _clip_reward(reward):
    return np.clip(reward, -1.0, 1.0)

"""
def _process_obs(obs):
    return obs  #np.reshape(obs, (1, obs.shape[0])) #24

def _clip_reward(reward):
    return reward / 10.0
"""

def _end_episode(envs, episode, data, max_step, episode_step):
    #global_step.assign_add(1)
    print('Episode: ', episode, ' entropy: ', data[0] / float(episode_step), ' reward', data[1], ' global_step: ', max_step, ' episode_step: ', episode_step)
    writer.add_scalar(envs.scenario + '/Entropy', data[0] / float(episode_step), episode)
    writer.add_scalar(envs.scenario + '/Reward', data[1], episode)
    writer.add_scalar(envs.scenario + '/Episode_step', episode_step, episode)
    writer.add_scalar(envs.scenario + '/Decay/learning_rate', data[2], episode)
    writer.add_scalar(envs.scenario + '/Decay/clip_param', data[3], episode)

def train(next_value, actorCriticOld, rollouts, arglist):
    values = []
    rewards = []
    masks = []
    actions = []
    log_probs = []
    states = []

    for rollout in rollouts:
        obs_d, actions_d, rewards_d, values_d, log_probs_d, terminals_d = rollout.get_storage()
        actions.append(actions_d)
        states.append(obs_d)
        rewards.append(rewards_d)
        values.append(values_d)
        log_probs.append(log_probs_d)
        masks.append(terminals_d)
    
    values = np.array(values)
    rewards = np.array(rewards)
    masks = np.array(masks)
    actions = np.array(actions)
    log_probs = np.array(log_probs)
    states = np.array(states)
    
    #print('next_value:', next_value.shape)
    #print('masks:', masks.shape)
    #print('rewards:', rewards.shape)
    
    returns = ppo.compute_returns(rewards, next_value, masks, arglist.gamma)

    #print('values:', values.shape)
    #print('returns:', returns.shape)
    advantage = ppo.compute_gae(rewards, values, next_value, masks, arglist.gamma, arglist.lam)
    #print('advantage:', advantage.shape)

    advantage = (advantage - np.mean(advantage)) / np.std(advantage)

    indices = np.random.permutation(range(arglist.time_horizon))
    states = states[:, indices]
    actions = actions[:, indices]
    log_probs = log_probs[:, indices]
    returns = returns[:, indices]
    advantage = advantage[:, indices]
    masks = masks[:, indices]

    ppo.update(actorCriticOld, arglist.time_horizon, arglist.epoch, arglist.batch_size, states, actions, log_probs, returns, advantage, arglist.grad_clip, arglist.value_factor, arglist.entropy_factor)

    for rollout in rollouts:
        rollout.flush()

    pass

def main(arglist):
    envs = env_wrapper.EnvWrapper(arglist.scenario, arglist.actors, arglist.saved_episode, update_obs=_process_obs, update_reward=_clip_reward, end_episode=_end_episode)
    rollouts = [roll.Rollout() for _ in range(arglist.actors)] #envs.observation_shape[0]
    #actorCriticOld = agent.ActorCritic(envs.action_shape, arglist.learning_rate, arglist.epsilon, arglist.final_step, envs.observation_shape, envs.continious, envs.upper_bound).to(device)
    actorCriticOld = agent.ActorCriticCNN(envs.action_shape, arglist.learning_rate, arglist.epsilon, arglist.final_step, [4, 84, 84], envs.continious, envs.upper_bound).to(device)
    try:
        actorCriticOld.load_model('./saved_actor/' + arglist.scenario + '_' + str(arglist.load_episode_saved))
    except Exception as e:
        print('FAILED TO LOAD.')

    t = 0
    batch_obs = envs.reset()

    batch_stack = []
    
    for obs in batch_obs:
        stack = np.concatenate([obs, obs], axis=1)
        stack = np.concatenate([stack, obs], axis=1)
        stack = np.concatenate([stack, obs], axis=1)
        batch_stack.append(stack)
    
    while t < arglist.final_step:
        if not arglist.eval:
            envs.render(0)

        actions_t = []
        values_t = []
        dist_cat_t = []
        entropy_t = []

        for stack in batch_stack:
            state = torch.FloatTensor(stack).to(device)
            #print('shape:', state.shape)
            dist, value = actorCriticOld(state)
            #print('value:', value)
            #print('action:', dist.sample())
            action = dist.sample()[0] #============================================>
            #print('action:', action)
            if envs.continious:
                action = action * envs.upper_bound
            entropy_t.append(dist.entropy().mean())
            actions_t.append(action.cpu().numpy()) #FIX
            #print('dist.log_prob(action).cpu().detach().numpy(): ', dist.log_prob(action).cpu().detach().numpy()) #============================================>
            dist_cat_t.append(dist.log_prob(action).cpu().detach().numpy()[0])
            #print('value:', value.cpu().detach().numpy())
            values_t.append(value.cpu().detach().numpy()[0][0]) #[0] FIX ============================================>

        obs2s_t, rewards_t, dones_t = envs.step(actions_t)

        for i in range(arglist.actors):
            data = envs.get_variables_at_index(i)
            if len(data) < 4:
                data = [0, 0, 0, 0]
            envs.add_variables_at_index(i, [entropy_t[i].cpu().detach().numpy() + data[0],
                                        rewards_t[i] + data[1],
                                        actorCriticOld.learning_rate_decay,
                                        actorCriticOld.clip_param])

        if t > 0 and (t / arglist.actors) % arglist.time_horizon == 0 and arglist.eval:
            next_values = np.reshape(values_t, [-1])
            train(next_values, actorCriticOld, rollouts, arglist)

        if arglist.eval:
            if envs.can_saved:
                actorCriticOld.save_model('./saved_actor/' + arglist.scenario + '_' + str(envs.episode))

        if arglist.eval:
            for i, rollout in enumerate(rollouts):
                rollout.add(batch_stack[i][0,:], actions_t[i], rewards_t[i], values_t[i], dist_cat_t[i], 1 - dones_t[i]) # 1 - ...
                #rollout.add(batch_obs[i], actions_t[i], rewards_t[i], values_t[i], dist_cat_t[i], 1 - dones_t[i]) # 1 - ...

        t += arglist.actors

        
        for i, stack in enumerate(batch_stack):
            stack = stack[:,1:,:,:]
            batch_stack[i] = np.concatenate([stack, obs2s_t[i]], axis=1)
        
        """
        for i, stack in enumerate(batch_obs):
            batch_obs[i] = obs2s_t[i]
        """

        if arglist.learning_rate_decay == 'linear' and arglist.eval:
            progress = t / arglist.final_step
            actorCriticOld.decay_clip_param(progress)
            actorCriticOld.decay_learning_rate(progress)

if __name__ == '__main__':
    arglist = parse_args()
    if arglist.eval == False:
        arglist.actors = 1
    main(arglist)
    print(arglist.scenario + ' Done.')