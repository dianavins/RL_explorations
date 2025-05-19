"""
- Asynchronous Actor Critic model using SNN
- Actor's actions are population coded
- Critic's reward is its membrane potential
"""

import torch
import torch.nn as nn
import snntorch as snn
import snntorch.functional as SF
from snntorch import surrogate
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical
import gymnasium as gym
import time
import matplotlib.pyplot as plt
import numpy as np
import collections

#torch.autograd.set_detect_anomaly(True)

# Hyperparameters
LEARNING_RATE = 0.0003
GAMMA = 0.99
MAX_EPISODES = 4000
T_MAX = 5
UPDATE_GLOBAL_ITER = 20
ENTROPY_BETA = 0.01
THRESHOLD = 0.2
SPIKE_STEPS = 100

# how many neurons per action class
NUM_NEURONS_PER_ACTION = 10

beta = 0.95

# Define surrogate gradient function at module level, not inside class
# This makes it picklable for multiprocessing


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

def atan_with_alpha(x):
    return surrogate.ATan.apply(x, 2.0)  # pass alpha explicitly

class Actor(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dims, 256)
        self.neuron1 = snn.Leaky(beta=beta, spike_grad=atan_with_alpha)
        self.fc2 = nn.Linear(256, n_actions*NUM_NEURONS_PER_ACTION)
        self.neuron2 = snn.Leaky(beta=beta, spike_grad=atan_with_alpha)
        self.n_actions = n_actions
        self.optimizer = optim.RMSprop(self.parameters(), lr=LEARNING_RATE, alpha=0.99, eps=1e-08)
        
    def forward(self, x, mem0):   
        cur1 = self.fc1(x)
        spk1, mem1 = self.neuron1(cur1, mem0)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.neuron2(cur2, mem1)
        return spk2, mem2
    
    def init_membranes(self):
        mem1 = torch.zeros(256*NUM_NEURONS_PER_ACTION).to(device)
        return mem1
    
class Critic(nn.Module):
    def __init__(self, input_dims):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dims, 256)
        self.neuron1 = snn.Leaky(beta=beta, spike_grad=atan_with_alpha)
        self.fc2 = nn.Linear(256, 1)
        self.neuron2 = snn.Leaky(beta=beta, spike_grad=atan_with_alpha)
        self.optimizer = optim.RMSprop(self.parameters(), lr=LEARNING_RATE, alpha=0.99, eps=1e-08)
        
    def forward(self, x, mem0):
        cur1 = self.fc1(x)
        spk1, mem1 = self.neuron1(cur1, mem0)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.neuron2(cur2, mem1)
        return spk2, mem2
    
    def init_membranes(self):
        mem1 = torch.zeros(128).to(device)
        return mem1

class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, SPIKE_STEPS=SPIKE_STEPS):
        super(ActorCritic, self).__init__()
        
        self.actor = Actor(input_dims, n_actions)
        
        # Critic (value) layers as spiking neurons
        self.critic = Critic(input_dims)
        
        # self.total_spikes = 0
        self.SPIKE_STEPS = SPIKE_STEPS  # Number of time steps for Poisson spike train
        self.n_actions = n_actions  # Number of actions
        
        # Move these outside __init__ to avoid unpicklable objects
        self.episode_reward = []  # Will be replaced by shared list in main()
        
    def normalize_observations(self, states):
        """Batch-compatible normalization for CartPole"""
        # states shape: (batch_size, 4)
        normalized = torch.zeros_like(states)
        
        # Known bounds (CartPole-v1 specific)
        POS_SCALE = 4.8 * 2  # -4.8 to 4.8 → 0 to 9.6
        ANGLE_SCALE = 0.42 * 2  # -0.42 to 0.42 → 0 to 0.84
        VEL_SCALE = 3.0  # Heuristic scaling factor for velocities
        
        # Position and angle (hard bounds)
        normalized[..., 0] = (states[..., 0] + 4.8) / POS_SCALE
        normalized[..., 2] = (states[..., 2] + 0.42) / ANGLE_SCALE
        
        # Velocities (soft bounds via sigmoid)
        normalized[..., 1] = torch.sigmoid(states[..., 1] / VEL_SCALE)
        normalized[..., 3] = torch.sigmoid(states[..., 3] / VEL_SCALE)
        
        return torch.clamp(normalized, 0, 1)  # Ensure no values exceed [0,1]
            
    def forward(self, x):
        # Ensure x is a tensor and has correct dimensions
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Ensure x is 2D (batch x features)
        if x.dim() == 1:
            x = x.unsqueeze(0)
              
        # Generate Poisson spike trains for entire batch
        poisson_spikes = snn.spikegen.rate(x, num_steps=self.SPIKE_STEPS).to(device)
        
        # Initialize containers for actor spikes sum
        spike_list = []
        
        # Reset membrane potentials at the beginning of each forward pass
        first_critic_membrane = self.critic.init_membranes()
        first_actor_membrane = self.actor.init_membranes()
        
        # Process all time steps
        for t in range(self.SPIKE_STEPS):
            # Get spikes for current time step (shape: batch_size x num_features)
            input_spikes = poisson_spikes[t, :, :]

            # Actor pathway
            actor_spikes, actor_membrane = self.actor(input_spikes, first_actor_membrane)
            
            # Critic pathway
            critic_spikes, critic_membrane = self.critic(input_spikes, first_critic_membrane)
            
            # Accumulate output spikes from actor network
            spike_list.append(actor_spikes.clone())
            
            # Update membrane potentials
            first_actor_membrane = actor_membrane  # Update membrane for next time step
            first_critic_membrane = critic_membrane  # Update membrane for next time step
    
        # Process action neuron groups to get policy distribution
        spike_tensor = torch.stack(spike_list, dim=0)
        actor_spikes_sum = spike_tensor.sum(dim=0)
        # Sum over first half of output neurons for action 1
        action_1_value = actor_spikes_sum[:, :NUM_NEURONS_PER_ACTION].sum(dim=1)
        # Sum over second half of output neurons for action 2
        action_2_value = actor_spikes_sum[:, NUM_NEURONS_PER_ACTION:].sum(dim=1)
        # Calculate total spikes
        action_values = torch.stack([action_1_value, action_2_value], dim=1)
        # Stack probabilities to form distribution
        policy_dists = F.softmax(action_values, dim=1)
        
        # Return policy distribution and critic value
        return policy_dists, critic_membrane
        
    def global_plot_reward_vs_epsode(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_reward)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Global Reward vs Episode')
        plt.show()
        # save as png
        plt.savefig('global_reward_vs_episode_A3CPopulation3.png')

class Worker(mp.Process):
    def __init__(self, global_net, optimizer, global_ep_idx, name, input_dims, n_actions):
        super(Worker, self).__init__()
        self.name = name
        self.local_net = ActorCritic(input_dims, n_actions)
        self.global_net = global_net
        self.optimizer = optimizer
        self.global_ep_idx = global_ep_idx
        
        self.env = None  # Initialize env in run() to avoid pickling issues
        self.input_dims = input_dims
        self.n_actions = n_actions
        
    def run(self):
        """Main process of each worker"""
        # Create environment here to avoid pickling issues
        self.env = gym.make('CartPole-v1')
        
        total_step = 1
        while self.global_ep_idx.value < MAX_EPISODES:
            state, _ = self.env.reset()
            buffer_state, buffer_action, buffer_value, buffer_reward = [], [], [], []
            episode_reward = 0
            
            for t in range(10000):  # max steps in an episode
                # Select action using spiking neural network
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs, value = self.local_net(state_tensor)
                m = Categorical(action_probs)
                action = m.sample().item()
                
                # Execute action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store experience
                buffer_state.append(state)
                buffer_action.append(action)
                buffer_reward.append(reward)
                buffer_value.append(value.item())  # Store scalar value instead of tensor
                
                episode_reward += reward
                
                # Update if it's time or episode done
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    self.update_global_network(buffer_state, buffer_action, buffer_value, buffer_reward, done, next_state)
                    buffer_state, buffer_action, buffer_value, buffer_reward = [], [], [], []
                    
                    # Sync with global network (pull)
                    self.local_net.load_state_dict(self.global_net.state_dict())
                
                state = next_state
                total_step += 1
                
                if done:
                    break
            
            # Record episode score
            with self.global_ep_idx.get_lock():
                self.global_ep_idx.value += 1
                self.global_net.episode_reward.append(episode_reward)  # Add to shared list
            
            print(f'Episode: {self.global_ep_idx.value}, Worker: {self.name}, Score: {episode_reward}')
            # self.local_net.total_spikes = 0  # Reset spikes for next episode
    
    def update_global_network(self, buffer_state, buffer_action, buffer_value, buffer_reward, done, next_state):
        """Update the global network using local gradients"""
        # Convert buffers to tensors
        states = torch.FloatTensor(np.array(buffer_state)).to(device)
        actions = torch.LongTensor(np.array(buffer_action)).view(-1, 1).to(device)
        values = torch.FloatTensor(np.array(buffer_value)).view(-1, 1).to(device)  # Already detached during storage
        rewards = torch.FloatTensor(np.array(buffer_reward)).view(-1, 1).to(device)
        
        # Calculate discounted returns
        if done:
            R = 0
        else:
            with torch.no_grad():
                R = self.local_net(torch.FloatTensor(next_state).unsqueeze(0))[1].detach().item()
        
        returns = []
        for r in buffer_reward[::-1]:
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).view(-1, 1).to(device)
        
        # Calculate advantages
        advantages = returns - values
        
        action_probs_list = []
        current_values_list = []

        # for i in range(states.size(0)):  # Loop over timesteps
        #     single_state = states[i].unsqueeze(0)  # Shape: [1, state_dim]
        #     probs, val = self.local_net(single_state)
        #     action_probs_list.append(probs)
        #     current_values_list.append(val)
        
        # batch process the states
        action_probs, current_values = self.local_net(states)
        
        
        action_probs_list.append(action_probs)
        current_values_list.append(current_values)

        # Stack to match expected shapes
        action_probs = torch.cat(action_probs_list, dim=0)          # Shape: [T, action_dim]

        
        # Create categorical distribution for actions
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions.squeeze())
        entropy = dist.entropy().mean()
        
        # Actor loss: -log_prob * advantage - entropy_bonus
        actor_loss = -(log_probs * advantages.detach().squeeze()).mean() - ENTROPY_BETA * entropy
        
        # Critic loss: MSE between critic's value estimates and returns
        critic_loss = F.mse_loss(current_values, returns)
        # else:
        #     # Handle case where current_values might be a scalar
        #     critic_loss = F.mse_loss(torch.tensor([current_values]).unsqueeze(-1), returns)
        
        # Zero gradients
        self.local_net.actor.optimizer.zero_grad()
        self.local_net.critic.optimizer.zero_grad()
        self.global_net.actor.optimizer.zero_grad()
        self.global_net.critic.optimizer.zero_grad()
        
        # ACTOR BACKPROP + UPDATES
        
        actor_loss.backward(retain_graph=True)  # Retain graph for multiple backward passes
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.local_net.parameters(), 40)
        
        # Transfer local gradients to global network
        for local_param, global_param in zip(self.local_net.actor.parameters(), self.global_net.actor.parameters()):
            if global_param.grad is None:
                global_param.grad = local_param.grad
            else:
                global_param.grad += local_param.grad
        
        # Update global network
        self.global_net.actor.optimizer.step()
        
        # copy global network parameters to local network
        for local_param, global_param in zip(self.local_net.actor.parameters(), self.global_net.actor.parameters()):
            local_param.data.copy_(global_param.data)
        
        # CRITIC BACKPROP + UPDATES
        
        critic_loss.backward(retain_graph=True)  # Retain graph for multiple backward passes
        
        # Transfer local gradients to global network
        for local_param, global_param in zip(self.local_net.critic.parameters(), self.global_net.critic.parameters()):
            if global_param.grad is None:
                global_param.grad = local_param.grad
            else:
                global_param.grad += local_param.grad
        
        # Update global network
        self.global_net.critic.optimizer.step()
        
        # copy global network parameters to local network
        for local_param, global_param in zip(self.local_net.critic.parameters(), self.global_net.critic.parameters()):
            local_param.data.copy_(global_param.data)


def main():
    # Set start method for multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Environment details
    env = gym.make('CartPole-v1')
    input_dims = env.observation_space.shape[0]  # 4 for CartPole
    n_actions = env.action_space.n  # 2 for CartPole
    env.close()
    
    # Create global network
    global_net = ActorCritic(input_dims, n_actions)
    global_net.share_memory()  # Share the global parameters in multiprocessing
    
    # Create manager for shared list
    manager = mp.Manager()
    global_net.episode_reward = manager.list()  # Shared list for rewards
    
    # Using RMSprop as mentioned in the A3C paper
    optimizer = optim.RMSprop(global_net.parameters(), lr=LEARNING_RATE, alpha=0.99, eps=1e-08)
    
    # Global counter for episodes
    global_ep_idx = mp.Value('i', 0)
    
    # Create workers (use fewer workers to reduce CPU load if needed)
    n_workers = min(mp.cpu_count(), 10)  # Limit to 4 workers max
    workers = [Worker(global_net, optimizer, global_ep_idx, f'w{i}', 
                     input_dims, n_actions) for i in range(n_workers)]
    
    # Start training
    [w.start() for w in workers]
    [w.join() for w in workers]
    
    # Convert shared list to regular list for plotting
    episode_rewards = list(global_net.episode_reward)
    global_net.episode_reward = episode_rewards
    
    # Plot global rewards
    global_net.global_plot_reward_vs_epsode()
    
    # Save the global network
    torch.save(global_net.state_dict(), 'global_network_A3CSNNPoisson.pth')


if __name__ == "__main__":
    main()
