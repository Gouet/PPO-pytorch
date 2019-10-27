import numpy as np
import torch
from torch.distributions import Categorical
from torch.distributions import Normal
#tf.enable_eager_execution()
import os

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def compute_returns(rewards, bootstrap_value, terminals, gamma):
    # (N, T) -> (T, N)
    rewards = np.transpose(rewards, [1, 0])
    terminals = np.transpose(terminals, [1, 0])
    returns = []
    R = bootstrap_value
    for i in reversed(range(rewards.shape[0])):
        R = rewards[i] + terminals[i] * gamma * R
        returns.append(R)
    returns = reversed(returns)
    # (T, N) -> (N, T)
    returns = np.transpose(list(returns), [1, 0])
    #returns = np.array(returns)
    return returns

def compute_gae(rewards, values, bootstrap_values, terminals, gamma, lam):
    # (N, T) -> (T, N)
    rewards = np.transpose(rewards, [1, 0])
    values = np.transpose(values, [1, 0])
    values = np.vstack((values, [bootstrap_values]))
    terminals = np.transpose(terminals, [1, 0])
    # compute delta
    deltas = []
    for i in reversed(range(rewards.shape[0])):
        V = rewards[i] + (terminals[i]) * gamma * values[i + 1]
        delta = V - values[i]
        deltas.append(delta)
    deltas = np.array(list(reversed(deltas)))
    # compute gae
    A = deltas[-1,:]
    advantages = [A]
    for i in reversed(range(deltas.shape[0] - 1)):
        A = deltas[i] + (terminals[i]) * gamma * lam * A
        advantages.append(A)
    advantages = reversed(advantages)
    # (T, N) -> (N, T)
    advantages = np.transpose(list(advantages), [1, 0])
    return advantages

"""
def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns
"""
def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = len(states)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[0, rand_ids, :], actions[0, rand_ids], log_probs[0, rand_ids], returns[0, rand_ids], advantage[0, rand_ids]

def _pick_batch(mini_batch_size, data, batch_index, flat=True, shape=None):
    start_index = batch_index * mini_batch_size
    batch_data = data[:, start_index:start_index + mini_batch_size]
    if flat:
        if shape is not None:
            return np.reshape(batch_data, [-1] + shape)
        return np.reshape(batch_data, [-1])
    else:
        return batch_data



def update(actorCritic, time_horizon, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, grad_clip=0.5, value_factor=1, entropy_factor=0.0):

    for epoch in range(ppo_epochs):
        for i in range(int(time_horizon / mini_batch_size)):
            index = i * mini_batch_size
            #print('actions.shape:', actions.shape)
            #print('log_probs.shape:', log_probs.shape)
            #print('states.shape:', states.shape)
            if actorCritic.continuous == False:
                batch_actions = _pick_batch(mini_batch_size, actions, i) #, shape=[actorCritic.num_action]
                batch_log_probs = _pick_batch(mini_batch_size, log_probs, i) #PENSE QUE C4EST PAS DU (64, num_action) , shape=[actorCritic.num_action]
            else:
                batch_actions = _pick_batch(mini_batch_size, actions, i, shape=[actorCritic.num_action]) #, shape=[actorCritic.num_action]
                batch_log_probs = _pick_batch(mini_batch_size, log_probs, i, shape=[actorCritic.num_action]) #PENSE QUE C4EST PAS DU (64, num_action) , shape=[actorCritic.num_action]
            
            #print('batch_actions.shape:', batch_actions.shape) #(64,)
            #print('batch_log_probs.shape:', batch_log_probs.shape) #(64,)

            #batch_obs = _pick_batch(mini_batch_size, states, i, shape=[actorCritic.state_shape[0]]) #[actorCritic.state_shape] FIX
            batch_obs = _pick_batch(mini_batch_size, states, i, shape=actorCritic.state_shape) #[actorCritic.state_shape] FIX
            batch_returns = _pick_batch(mini_batch_size, returns, i)
            batch_advs = _pick_batch(mini_batch_size, advantages, i)

            state = torch.FloatTensor(batch_obs).to(device)

            #print('state.shape: ', state.shape)
            dist, value = actorCritic(state)

            #print('ACTIONS: ', dist.sample().shape)
            #print('value: ', value.shape)

            batch_returns = np.reshape(batch_returns, (batch_returns.shape[0], 1))
            batch_returns = torch.FloatTensor(batch_returns).to(device)

            batch_advs = torch.FloatTensor(batch_advs).to(device)


            #print('batch_returns', batch_returns.shape)

            #print('batch_advs.shape:', batch_advs.shape)

            value_loss = (batch_returns - value).pow(2).mean()
            value_loss *= value_factor
            
            entropy = dist.entropy().mean()
            entropy *= entropy_factor

            #print('BEFORE batch_actions:', batch_actions.shape)
            batch_actions = torch.FloatTensor(batch_actions).to(device)

            batch_log_probs = torch.FloatTensor(batch_log_probs).to(device)
            
            #print('AFTER batch_actions: ', batch_actions.shape)
            log_prob = dist.log_prob(batch_actions)
            #print('log_prob:', log_prob)
            #print('log_prob:', log_prob.shape) #(1, 64)    J'AI (64,64) FAUX
            #print('batch_log_probs: ', batch_log_probs.shape)
            
            ratio = (log_prob - batch_log_probs).exp()
            
            #print('ratio:', ratio)
            #print('ratio.shape:', ratio.shape) #(1,64)
            
            if actorCritic.continuous:
                ratio = ratio.mean(dim=1, keepdim=True)#tf.reduce_mean(ratio, axis=1, keep_dims=True) #TODO GARDER
            #else:
            #    ratio = ratio.view(-1, 1)#.reshape(ratio, [-1, 1])

            #print('AFTER ratio.shape:', ratio.shape)
            #print('AFTER batch_advs.shape:', batch_advs.shape)

            surr1 = ratio * batch_advs

            #print('surr1: ', surr1.shape)

            surr2 = torch.clamp(ratio, 1.0 - actorCritic.clip_param, 1.0 + actorCritic.clip_param) * batch_advs
            #print('surr2: ', surr2.shape)

            policy_loss = torch.min(surr1, surr2).mean()

            loss = value_loss - policy_loss - entropy

            actorCritic.optimizer.zero_grad()
            loss.backward()
            actorCritic.optimizer.step()




"""
class ActorCritic(tf.keras.Model):
    def __init__(self, num_action, lr, clip_param, final_step, state_shape, continuous=False, upper_bound=0, filename='./saved/actor.h5'):
        super(ActorCritic, self).__init__()
        self.filename = filename

        #w = tf.orthogonal_initializer(np.sqrt(2.0))
        self.upper_bound = upper_bound
        #self.input = tf.keras.layers.Input(shape=(6,1))
        self.state_shape = state_shape
        self.num_action = num_action
        self.continuous = continuous
        #self.input = tf.keras.layers.Input(shape=state_shape)

        self._lr = tf.Variable(lr)
        self.initial_learning_rate_value = lr
        self.final_learning_rate_step = final_step

        self.clip_param = clip_param
        self.initial_clip_param = clip_param
        self.final_clip_param_step = final_step

        activation = tf.nn.relu
        if self.continuous:
            activation = tf.nn.tanh

        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(150, activation=activation)
        self.d2 = tf.keras.layers.Dense(120, activation=activation)

        self.policy = tf.keras.layers.Dense(num_action)
        self.value = tf.keras.layers.Dense(1)
        def test(x):
            dist = tfd.Categorical(tf.nn.softmax(x))
            action = dist.sample(1)[0]
            return action, dist.log_prob(action), dist.entropy(), x

        self.policy2 = tf.keras.layers.Lambda(test)
        #self.policy3 = tf.keras.layers.Lambda(test2)
        #self.policy4 = tf.keras.layers.Lambda(test3)

                                               self.policy2 = tfp.layers.DistributionLambda(make_distribution_fn=lambda t:  tfd.Categorical(tf.nn.softmax(t)),
                                               convert_to_tensor_fn=lambda s: s.sample(1))
        x = tf.keras.layers.Flatten()(self.input)
        x = tf.keras.layers.Dense(150, activation=activation, kernel_initializer=w)(x)
        x = tf.keras.layers.Dense(120, activation=activation, kernel_initializer=w)(x)

        

        policy = tf.keras.layers.Dense(num_action, kernel_initializer=tf.orthogonal_initializer(0.1))(x)

        value = tf.keras.layers.Dense(1, kernel_initializer=tf.orthogonal_initializer(0.1))(x)
        policy = tfp.layers.DistributionLambda(make_distribution_fn=lambda t:  tfd.Categorical(tf.nn.softmax(t)),
                                               convert_to_tensor_fn=lambda s: s.sample(1))(policy)
        #policy = tf.distributions.Categorical(probs=tf.nn.softmax(policy))

        self.model = tf.keras.Model(inputs=[self.input], outputs=[policy, value])

        self.model.summary()


        #self.optimizer = tf.train.AdamOptimizer(self.lr, epsilon=1e-5)
        self.optimizer_lol = tf.keras.optimizers.Adam(lr, epsilon=1e-5)

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        y = self.policy(x)
        return [self.policy2(y), self.value(x)]

    def predict(self, states):
        distCat, value = self.predict(states)
        if self.continuous == False:
            #distCat = tfd.Categorical(probs=tf.nn.softmax(distCat))
            action = distCat.sample(1)[0]
        else:
            std = tf.zeros_like(distCat) + tf.exp(np.zeros((1, self.num_action), dtype=np.float32))
            distCat = tfd.Normal(loc=distCat, scale=std)
            action = distCat.sample(1)[0]
        return distCat, value, action

    def __call__(self, states):
        distCat, value = self.model(states)
        if self.continuous == False:
            pass
            distCat = tfd.Categorical(probs=tf.nn.softmax(distCat))
        else:
            std = tf.zeros_like(distCat) + tf.exp(np.zeros((1, self.num_action), dtype=np.float32))
            distCat = tfd.Normal(loc=distCat, scale=std)
        return distCat, value

    def save(self, directory='', filename=''):
        if filename != '' and directory != '':
            try:  
                os.mkdir(directory)
                self.save_weights(directory + filename)
            except OSError:  
                print ("Creation of the directory failed")
            else:  
                print ("Successfully created the directory")
            
        else:
            self.save_weights(self.filename)

    def decay_learning_rate(self, step):
        decay = 1.0 - (float(step) / self.final_learning_rate_step)
        if decay < 0.0:
            decay = 0.0
        self.lr.assign(decay * self.initial_learning_rate_value)

    def decay_clip_param(self, step):
        decay = 1.0 - (float(step) / self.final_clip_param_step)
        if decay < 0.0:
            decay = 0.0
        self.clip_param = (decay * self.initial_clip_param)

    def load(self, filename=''):
        if filename != '':
            self.load_weights(filename)
        else:   
            self.load_weights(self.filename)
    
    def hard_copy(self, actor_var):
        [self.trainable_variables[i].assign(actor_var[i])
                for i in range(len(self.trainable_variables))]

"""