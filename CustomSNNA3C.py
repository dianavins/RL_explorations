import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical
import gymnasium as gym
import time
import matplotlib.pyplot as plt

"""
- Asynchronous Actor Critic model using custom spiking neuron classes
- using Gymnasium's Cartpole
"""

# Hyperparameters
LEARNING_RATE = 0.0003
GAMMA = 0.99
MAX_EPISODES = 2000
T_MAX = 5
UPDATE_GLOBAL_ITER = 5
ENTROPY_BETA = 0.01
THRESHOLD = 0.8

# STDP Learning Parameters
STDP_LEARNING_RATE = 0.01
TRACE_DECAY = 0.9

class SpikingNeuron(nn.Module):
    def __init__(self, input_size, output_size, threshold=THRESHOLD, decay_rate=0.9):
        super(SpikingNeuron, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Synaptic weights
        self.weights = nn.Parameter(torch.randn(output_size, input_size) * 0.1)
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(output_size))
        
        # Neuron parameters
        self.threshold = threshold
        self.decay_rate = decay_rate
        
        # STDP-related traces
        self.pre_traces = None
        self.post_traces = None
        
    def forward(self, x, prev_membrane_potential=None):
        """
        Forward pass for spiking neurons
        x: input
        prev_membrane_potential: previous membrane potential for continuous learning
        """
        # Ensure x is a tensor and has correct shape
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Flatten input if it's a multi-dimensional tensor
        x = x.view(-1)
        
        # Ensure input size matches weights
        if x.size(0) != self.input_size:
            # Pad or truncate input to match expected size
            if x.size(0) < self.input_size:
                x = F.pad(x, (0, self.input_size - x.size(0)))
            else:
                x = x[:self.input_size]
        
        # Compute membrane potential
        membrane_potential = F.linear(x.unsqueeze(0), self.weights, self.bias)
        
        # Generate spikes based on threshold
        spikes = (membrane_potential >= self.threshold).float()
        
        # Initialize traces if None
        if self.pre_traces is None:
            self.pre_traces = torch.zeros_like(x)
            self.post_traces = torch.zeros_like(spikes.squeeze())
        
        # Decay traces
        self.pre_traces = self.pre_traces * TRACE_DECAY
        self.post_traces = self.post_traces * TRACE_DECAY
        
        # Update traces
        self.pre_traces[x > 0] = 1.0
        self.post_traces[spikes.squeeze() > 0] = 1.0
        
        # total number of spikes
        # total_spikes = torch.sum(spikes)
        
        return spikes.squeeze(), membrane_potential.squeeze()

    def stdp_update(self, pre_spikes, post_spikes):
        """
        Spike-Timing-Dependent Plasticity (STDP) weight update
        """
        # Ensure pre_spikes and post_spikes are 1D tensors
        pre_spikes = pre_spikes.float().view(-1)
        post_spikes = post_spikes.float().view(-1)
        
        # Truncate or pad if needed
        if pre_spikes.size(0) > self.input_size:
            pre_spikes = pre_spikes[:self.input_size]
        if post_spikes.size(0) > self.output_size:
            post_spikes = post_spikes[:self.output_size]        
        
        # Compute weight changes based on pre and post synaptic spikes
        # Potentiation (post before pre): outer(post, pre)
        potentiation = torch.outer(post_spikes, pre_spikes)

        # Depression (pre before post): need to transpose the outer product
        depression = torch.outer(pre_spikes, post_spikes).T  # Transpose to match weight matrix shape

        delta_weights = STDP_LEARNING_RATE * (potentiation - depression)
        
        # Update weights
        with torch.no_grad():
            self.weights += delta_weights

class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(ActorCritic, self).__init__()
        
        # Shared layers as spiking neurons
        self.shared_layer = SpikingNeuron(input_dims, 128)
        
        # Actor (policy) layers as spiking neurons
        self.actor_layer1 = SpikingNeuron(128, 64)
        self.actor_layer2 = SpikingNeuron(64, n_actions)
        
        # Critic (value) layers as spiking neurons
        self.critic_layer1 = SpikingNeuron(128, 64)
        self.critic_layer2 = SpikingNeuron(64, 1)
        
        self.total_spikes = 0
        self.episode_reward = []  # Store rewards for plotting
        
    
    def forward(self, x):
        # Ensure x is a tensor and has correct dimensions
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Ensure x is 2D (batch x features)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Process each state in the batch
        policy_dists = []
        values = []
        
        for state in x:
            
            # Shared layer
            shared_spikes, shared_membrane = self.shared_layer(state)
            
            # Actor pathway
            actor_spikes1, actor_membrane1 = self.actor_layer1(shared_spikes)
            actor_spikes2, actor_membrane2 = self.actor_layer2(actor_spikes1)
            
            # Convert actor spikes to policy distribution
            policy_dist = F.softmax(actor_membrane2, dim=-1)
            
            # Critic pathway
            critic_spikes1, critic_membrane1 = self.critic_layer1(shared_spikes)
            critic_spikes2, critic_membrane2 = self.critic_layer2(critic_spikes1)
            
            policy_dists.append(policy_dist)
            values.append(critic_membrane2)
            
            total_spikes = shared_spikes.sum() + actor_spikes1.sum() + actor_spikes2.sum() + critic_spikes1.sum() + critic_spikes2.sum()
            self.total_spikes += total_spikes.item()
        
        # Stack results
        policy_dists = torch.stack(policy_dists)
        values = torch.stack(values)
        
        return policy_dists, values

    def perform_stdp(self, input_spikes):
        """
        Perform STDP learning across all layers
        """
        # Ensure input_spikes is a tensor
        if not isinstance(input_spikes, torch.Tensor):
            input_spikes = torch.tensor(input_spikes, dtype=torch.float32)
        
        # Shared layer STDP
        shared_spikes, _ = self.shared_layer(input_spikes)
        
        # Actor pathway STDP
        actor_spikes1, _ = self.actor_layer1(shared_spikes)
        actor_spikes2, _ = self.actor_layer2(actor_spikes1)
        
        # Critic pathway STDP
        critic_spikes1, _ = self.critic_layer1(shared_spikes)
        critic_spikes2, _ = self.critic_layer2(critic_spikes1)
        
        # Perform STDP updates
        self.shared_layer.stdp_update(input_spikes, shared_spikes)
        self.actor_layer1.stdp_update(shared_spikes, actor_spikes1)
        self.actor_layer2.stdp_update(actor_spikes1, actor_spikes2)
        self.critic_layer1.stdp_update(shared_spikes, critic_spikes1)
        self.critic_layer2.stdp_update(critic_spikes1, critic_spikes2)
        
    def global_plot_reward_vs_epsode(self):
        print("self.episode_reward", self.episode_reward)
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_reward)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Global Reward vs Episode')
        plt.show()
        # save as png
        plt.savefig('global_reward_vs_episode_A3CSNN.png')

class Worker(mp.Process):
    def __init__(self, global_net, optimizer, global_ep_idx, name, input_dims, n_actions):
        super(Worker, self).__init__()
        self.name = name
        self.local_net = ActorCritic(input_dims, n_actions)
        self.global_net = global_net
        self.optimizer = optimizer
        self.global_ep_idx = global_ep_idx
        
        self.env = gym.make('CartPole-v1')
        self.input_dims = input_dims
        self.n_actions = n_actions
        
    def run(self):
        """Main process of each worker"""
        total_step = 1
        while self.global_ep_idx.value < MAX_EPISODES:
            state, _ = self.env.reset()
            buffer_state, buffer_action, buffer_reward = [], [], []
            episode_reward = 0
            
            for t in range(10000):  # max steps in an episode
                # Select action using spiking neural network
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs, _ = self.local_net(state_tensor)
                m = Categorical(action_probs)
                action = m.sample().item()
                
                # Execute action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store experience
                buffer_state.append(state)
                buffer_action.append(action)
                buffer_reward.append(reward)
                
                episode_reward += reward
                
                # Update if it's time or episode done
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    self.update_global_network(buffer_state, buffer_action, buffer_reward, done, next_state)
                    buffer_state, buffer_action, buffer_reward = [], [], []
                    
                    # Sync with global network (pull)
                    self.local_net.load_state_dict(self.global_net.state_dict())
                
                state = next_state
                total_step += 1
                
                if done:
                    break
            
            # Record episode score
            with self.global_ep_idx.get_lock():
                self.global_ep_idx.value += 1
            
            print(f'Episode: {self.global_ep_idx.value}, Worker: {self.name}, Score: {episode_reward}, Total Spikes: {self.local_net.total_spikes}')
            self.local_net.total_spikes = 0  # Reset spikes for next episode
            self.global_net.episode_reward.append(episode_reward)
    
    def update_global_network(self, buffer_state, buffer_action, buffer_reward, done, next_state=None):
        """Update the global network using local gradients and STDP"""
        # If episode not done, bootstrap value
        R = 0
        if not done and next_state is not None:
            state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            _, value = self.local_net(state_tensor)
            R = value.detach().item()
        
        # Compute returns (discounted rewards)
        buffer_return = []
        for r in buffer_reward[::-1]:  # reverse buffer
            R = r + GAMMA * R
            buffer_return.insert(0, R)
        
        buffer_return = torch.FloatTensor(buffer_return)
        buffer_state = torch.FloatTensor(np.array(buffer_state))
        buffer_action = torch.LongTensor(buffer_action)
        
        # Forward pass
        action_probs, values = self.local_net(buffer_state)
        
        # Calculate advantages
        values = values.squeeze()
        advantages = buffer_return - values.detach()
        
        # Calculate policy loss
        m = Categorical(action_probs)
        log_probs = m.log_prob(buffer_action)
        entropy = m.entropy().mean()  # Encourage exploration
        
        # Actor loss: -log_prob * advantage - entropy_bonus
        actor_loss = -(log_probs * advantages).mean() - ENTROPY_BETA * entropy
        
        # Critic loss: MSE of returns vs predicted values
        critic_loss = F.mse_loss(values, buffer_return)
        
        # Perform STDP learning
        for state in buffer_state:
            self.local_net.perform_stdp(state)
        
        # Zero gradients before backward pass
        self.optimizer.zero_grad()
        
        # Calculate gradients separately for actor and critic and update global network
        actor_loss.backward(retain_graph=True)  
        critic_loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.local_net.parameters(), 40)
        
        # Push local gradients to global network
        for local_param, global_param in zip(
                self.local_net.parameters(), self.global_net.parameters()):
            if global_param.grad is None:
                global_param._grad = local_param.grad
            else:
                global_param._grad += local_param.grad  # Accumulate gradients if they exist
        
        # Apply gradients to global network
        self.optimizer.step()


def main():
    # Environment details
    env = gym.make('CartPole-v1')
    input_dims = env.observation_space.shape[0]  # 4 for CartPole
    n_actions = env.action_space.n  # 2 for CartPole
    env.close()
    
    # Create global network and optimizer
    global_net = ActorCritic(input_dims, n_actions)
    global_net.share_memory()  # Share the global parameters in multiprocessing
    
    # Use a manager to create a shared list for rewards
    manager = mp.Manager()
    global_net.episode_reward = manager.list()  # Shared list for rewards
    
    # Using RMSprop as mentioned in the A3C paper
    optimizer = optim.RMSprop(global_net.parameters(), lr=LEARNING_RATE, alpha=0.99, eps=1e-08)
    
    # Global counter for episodes
    global_ep_idx = mp.Value('i', 0)
    
    # Create workers (use fewer workers to reduce CPU load if needed)
    n_workers = min(mp.cpu_count(), 4)  # Limit to 4 workers max
    workers = [Worker(global_net, optimizer, global_ep_idx, f'w{i}', 
                     input_dims, n_actions) for i in range(n_workers)]
    
    # Start training
    [w.start() for w in workers]
    [w.join() for w in workers]
    
    # Plot global rewards
    global_net.global_plot_reward_vs_epsode()
    # Save the global network
    torch.save(global_net.state_dict(), 'global_network_A3CSNN.pth')


if __name__ == "__main__":
    # Use mp.set_start_method if on MacOS
    # mp.set_start_method('spawn')
    main()
