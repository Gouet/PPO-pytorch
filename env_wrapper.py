import gym

import torch
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

"""
    FROM https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    Thanks to OpenAI.
"""
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

"""
    FROM https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    Thanks to OpenAI.
"""
class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

"""
    FROM https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    Thanks to OpenAI.
"""
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class EnvWrapper:
    def __init__(self, gym_env, actors, saved_episode, update_obs=None, update_reward=None, end_episode=None):
        self.envs = []
        self.variables = []
        self.update_obs = update_obs
        self.episode = 0
        self.end_episode = end_episode
        self.update_reward = update_reward
        self.saved_episode = saved_episode
        self.global_step = 0
        self.episode_step = []
        self.can_saved = False
        self.scenario = gym_env
        for _ in range(actors):
            env = gym.make(gym_env)
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env)
            env = EpisodicLifeEnv(env)
            
            self.observation_shape = env.observation_space.shape
            if isinstance(env.action_space, gym.spaces.Box):
                self.action_shape = env.action_space.shape[0]
                self.upper_bound = torch.FloatTensor(env.action_space.high).to(device)
                self.continious = True
            else:
                self.action_shape = env.action_space.n
                self.upper_bound = 0
                self.continious = False
            self.envs.append(env)
        for _ in range(actors):
            self.variables.append([])
            self.episode_step.append(0)

    def add_variables_at_index(self, id, data):
        self.variables[id] = data

    def get_variables_at_index(self, id):
        return self.variables[id]

    def step(self, actions):
        batch_states = []
        batch_rewards = []
        batch_dones = []
        self.can_saved = False

        for i, action in enumerate(actions):
            self.episode_step[i] += 1
            states, rewards, done_, _ = self.envs[i].step(action) # action
            if done_ == True:
                states = self.envs[i].reset()
                self.episode += 1
                if self.episode % self.saved_episode == 0:
                    self.can_saved = True
                if self.end_episode is not None:
                    self.end_episode(self, self.episode, self.variables[i], self.global_step, self.episode_step[i])
                self.episode_step[i] = 0
                self.variables[i] = []
            if self.update_reward is not None:
                rewards = self.update_reward(rewards)
            if self.update_obs is not None:
                states = self.update_obs(states)
            batch_states.append(states)
            batch_rewards.append(rewards)
            batch_dones.append(done_)
        self.dones = batch_dones
        self.global_step += 1
        return batch_states, batch_rewards, batch_dones

    def render(self, id):
        self.envs[id].render()

    def done(self):
        return all(self.dones)

    def reset(self):
        batch_states = []
        self.dones = []
        print('RESET')
        for env in self.envs:
            obs = env.reset()
            self.dones.append(False)
            if self.update_obs is not None:
                obs = self.update_obs(obs)
            batch_states.append(obs)
        return batch_states