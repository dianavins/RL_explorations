"""
- population coded actor
- membrane coded critic
- weight updates are reward modulated STDP
- applied on Gymnasium Cartpole
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
MAX_EPISODES = 1000
ENTROPY_BETA = 0.01
THRESHOLD = 0.2
SPIKE_STEPS = 100
TIMESTEPS = 100

# how many neurons per action class
NUM_NEURONS_PER_ACTION = 10
STDP_LR = 0.0001

beta = 0.95

# Define surrogate gradient function at module level, not inside class
# This makes it picklable for multiprocessing


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

def atan_with_alpha(x):
    return surrogate.ATan.apply(x, 2.0)  # pass alpha explicitly

def apply_rstdp(TDE, layer, pre_spikes, post_spikes, tau=20.0):
    """
    Apply reward-modulated STDP update using spike time differences.
    
    Args:
        TDE (float): temporal difference error signal (reward prediction error)
        layer (nn.Linear): layer to update
        pre_spikes (Tensor): shape [pre_neurons, time]
        post_spikes (Tensor): shape [post_neurons, time]
        tau (float): decay time constant for STDP
    """
    pre_spikes = pre_spikes.detach()  # [pre_neurons, time]
    post_spikes = post_spikes.detach()  # [post_neurons, time]
    time_steps = TIMESTEPS # number of time steps

    # Initialize weight change accumulator
    dw = torch.zeros_like(layer.weight)

    for t_post in range(time_steps):
        for t_pre in range(time_steps):
            delta_t = t_post - t_pre
            if delta_t > 0:
                # Potentiation (pre before post)
                timing_scale = torch.exp(torch.tensor(-delta_t / tau, device=layer.weight.device))
                contribution = torch.ger(post_spikes[:, t_post], pre_spikes[:, t_pre]) * timing_scale
                dw += contribution
            elif delta_t < 0:
                # Depression (post before pre)
                timing_scale = torch.exp(torch.tensor(delta_t / tau, device=layer.weight.device))  # negative delta_t
                contribution = -torch.ger(post_spikes[:, t_post], pre_spikes[:, t_pre]) * timing_scale
                dw += contribution
            # delta_t == 0: no contribution

    dw *= TDE  # modulate by temporal difference error

    with torch.no_grad():
        layer.weight += STDP_LR * dw



class Actor(nn.Module):
    def __init__(self, input_dims, n_actions, time_steps=TIMESTEPS):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dims, 256)
        self.neuron1 = snn.Leaky(beta=beta, spike_grad=atan_with_alpha)
        self.fc2 = nn.Linear(256, n_actions * NUM_NEURONS_PER_ACTION)
        self.neuron2 = snn.Leaky(beta=beta, spike_grad=atan_with_alpha)
        self.n_actions = n_actions
        self.optimizer = optim.RMSprop(self.parameters(), lr=LEARNING_RATE, alpha=0.99, eps=1e-08)
        
        # tensor for outputs of both layers, size [2, n_actions * NUM_NEURONS_PER_ACTION, time_steps]
        self.neuron1_history = torch.zeros(2, 256, time_steps)
        self.neuron2_history = torch.zeros(2, n_actions * NUM_NEURONS_PER_ACTION, time_steps)

    def forward(self, x, mem0, t):   
        cur1 = self.fc1(x)
        spk1, mem1 = self.neuron1(cur1, mem0)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.neuron2(cur2, mem1)
        
        self.neuron1_history[0, :, t] = spk1.detach()
        self.neuron2_history[1, :, t] = spk2.detach()

        return spk2, mem2

    def init_membranes(self):
        return torch.zeros(256).to(device)
    
class Critic(nn.Module):
    def __init__(self, input_dims, time_steps=TIMESTEPS):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dims, 256)
        self.neuron1 = snn.Leaky(beta=beta, spike_grad=atan_with_alpha)
        self.fc2 = nn.Linear(256, 1)
        self.neuron2 = snn.Leaky(beta=beta, spike_grad=atan_with_alpha)
        self.optimizer = optim.RMSprop(self.parameters(), lr=LEARNING_RATE, alpha=0.99, eps=1e-08)
        self.neuron1_history = torch.zeros(2, 256, time_steps)
        self.neuron2_history = torch.zeros(2, 1, time_steps)

        self.neuron2_history = torch.zeros(2, 1, time_steps)

    def forward(self, x, mem0, t):
        cur1 = self.fc1(x)
        spk1, mem1 = self.neuron1(cur1, mem0)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.neuron2(cur2, mem1)

        # tensor for outputs of both layers, size [2, n_actions * NUM_NEURONS_PER_ACTION, time_steps]
        self.neuron1_history[0, :, t] = spk1.detach()
        self.neuron2_history[1, :, t] = spk2.detach()

        return spk2, mem2

    def init_membranes(self):
        return torch.zeros(256).to(device)

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
            
    def forward(self, poisson_spikes):
        
        # Reset membrane potentials at the beginning of each forward pass
        first_critic_membrane = self.critic.init_membranes()
        first_actor_membrane = self.actor.init_membranes()
        
        # Process all time steps
        for t in range(self.SPIKE_STEPS):
            # Get spikes for current time step (shape: batch_size x num_features)
            input_spikes = poisson_spikes[t, :, :]

            # Actor pathway
            actor_spikes, actor_membrane = self.actor(input_spikes, first_actor_membrane, t)
            
            # Critic pathway
            critic_spikes, critic_membrane = self.critic(input_spikes, first_critic_membrane, t)
            
            # Update membrane potentials
            first_actor_membrane = actor_membrane  # Update membrane for next time step
            first_critic_membrane = critic_membrane  # Update membrane for next time step        
        
        # TODO: sum actor's last layer spikes from self.actor.history to see total spikes per neuron
        actor_spikes_sum = self.actor.neuron2_history[1].sum(dim=1)
        # Sum over first half of output neurons for action 1
        action_1_value = actor_spikes_sum[:NUM_NEURONS_PER_ACTION].sum()
        # Sum over second half of output neurons for action 2
        action_2_value = actor_spikes_sum[NUM_NEURONS_PER_ACTION:].sum()
        # Calculate total spikes
        action_values = torch.stack([action_1_value, action_2_value])
        # Stack probabilities to form distribution
        policy_dists = F.softmax(action_values, dim=0)
        
        # Return policy distribution and critic value
        return policy_dists, critic_membrane
        
    def plot_reward_vs_epsode(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_reward)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward vs Episode')
        plt.show()
        # save as png
        plt.savefig('reward_vs_episode_actor-critic-r-stdp.png')



def main():
    
    # Environment details
    env = gym.make('CartPole-v1')
    input_dims = env.observation_space.shape[0]  # 4 for CartPole
    n_actions = env.action_space.n  # 2 for CartPole
    env.close()
    
    # Create  network
    net = ActorCritic(input_dims, n_actions)
    net.share_memory()  # Share the global parameters in multiprocessing
    
    # Using RMSprop as mentioned in the A3C paper
    optimizer = optim.RMSprop(net.parameters(), lr=LEARNING_RATE, alpha=0.99, eps=1e-08)
    
    # TODO: training loop
    for episode in range(MAX_EPISODES):
        state, info = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        done = False
        total_reward = 0
        while not done:
            
            # Generate Poisson spike trains for entire batch
            poisson_spikes = snn.spikegen.rate(state_tensor, num_steps=SPIKE_STEPS).to(device)
            
            # Get action probabilities and critic value
            policy_dists, critic_value = net(poisson_spikes)
            
            # Sample action from the policy distribution
            m = Categorical(policy_dists)
            action = m.sample()
            
            # Take action in the environment
            next_state, reward, terminated, truncated, info = env.step(action.item())
            
            done = terminated or truncated
            
            # update TDE
            TDE = reward - critic_value.item()
            
            # apply r-stdp
            apply_rstdp(TDE, net.actor.fc1, poisson_spikes[:, 0, :].T, net.actor.neuron1_history[0])
            apply_rstdp(TDE, net.actor.fc2, net.actor.neuron1_history[0], net.actor.neuron2_history[1])
            apply_rstdp(TDE, net.critic.fc1, poisson_spikes[:, 0, :].T, net.critic.neuron1_history[0])
            apply_rstdp(TDE, net.critic.fc2, net.critic.neuron1_history[0], net.critic.neuron2_history[1])
                
            # Update total reward
            total_reward += reward
            
            # Update state
            state = next_state
            
        # Store the total reward for this episode
        net.episode_reward.append(total_reward)
        
        if episode % UPDATE_GLOBAL_ITER == 0:
            # Perform global update here (e.g., apply STDP)
            pass
        
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")
    
    # Plot global rewards
    net.global_plot_reward_vs_epsode()
    
    # Save the global network
    torch.save(net.state_dict(), 'global_network_actor-critic-r-stdp.pth')


if __name__ == "__main__":
    main()
