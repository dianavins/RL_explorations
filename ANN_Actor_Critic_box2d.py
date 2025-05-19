import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt

"""
ANN Actor Critic for Gymnasium box2d environments
"""

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Mean and log_std for the Gaussian policy
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, state):
        # Actor
        x = self.actor(state)
        mean = torch.tanh(self.mean(x))  # Tanh constrains mean to [-1, 1]
        
        # Create the Normal distribution with learned mean and std
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        
        # Sample an action from the distribution
        action = dist.sample()
        
        # Clamp the action to be within [-1, 1]
        action = torch.clamp(action, -1.0, 1.0)
        
        # Calculate log probability of the action
        # To account for the tanh transformation, we need some adjustments
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Get value estimate
        value = self.critic(state)
        
        return action, log_prob, value
    
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            action, log_prob, _ = self.forward(state)
        
        return action.squeeze().cpu().numpy(), log_prob

# Training function
def train(env, model, optimizer, gamma=0.99, max_episodes=1000):
    episode_rewards = []
    
    for episode in range(max_episodes):
        state = env.reset()
        if isinstance(state, tuple):  # Handle newer gym API
            state = state[0]
        done = False
        episode_reward = 0
        
        # Lists to hold episode data
        log_probs = []
        values = []
        rewards = []
        masks = []
        
        while not done:
            # Select action
            actions, log_prob = model.act(state)
            # print("action dtype:", type(actions))
            # print("action:", actions)
            
            # Take action
            next_state, reward, done, _ = env.step(actions)[:4]
            
            # Store transition
            state_tensor = torch.FloatTensor(state)
            _, _, value = model(state_tensor) # returns action, log_prob, value
            
            log_probs.append(log_prob)
            values.append(value.squeeze())
            rewards.append(reward)
            masks.append(1-done)
            
            state = next_state
            episode_reward += reward
            
            # If episode is done, update model
            if done:
                # Calculate returns and advantages
                returns = []
                discounted_reward = 0
                
                for reward, mask in zip(reversed(rewards), reversed(masks)):
                    discounted_reward = reward + gamma * discounted_reward * mask
                    returns.insert(0, discounted_reward)
                
                returns = torch.FloatTensor(returns)
                
                # Convert lists to tensors
                # print("log_probs: ", log_probs)
                log_probs = torch.stack(log_probs)
                values = torch.stack(values)
                
                # Normalize returns (optional)
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                
                # Calculate advantages
                advantages = returns - values.detach()
                
                # Calculate losses
                actor_loss = -(log_probs * advantages).mean()  # Policy gradient loss
                critic_loss = F.mse_loss(values, returns)      # Value function loss
                
                # Total loss
                loss = actor_loss + 0.5 * critic_loss
                
                # Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Track episode rewards
        episode_rewards.append(episode_reward)
        
        # Print progress every 10 episodes
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Episode {episode+1}/{max_episodes}, Avg Reward (last 20): {avg_reward:.2f}")
        
        # Early stopping if solved (CartPole is considered solved at 195.0)
        if np.mean(episode_rewards[-100:]) >= 195.0 and len(episode_rewards) >= 100:
            print(f"Environment solved in {episode+1} episodes!")
            break
    
    return episode_rewards

# Main function
def main():
    # Create environment
    env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="rgb_array") # max reward = 300
    # env = gym.make("CarRacing-v3") # max reward = 900
    
    # Initialize model
    input_dim = env.observation_space.shape[0]  # for BipedalWalker-v3, input_dim = 24
    print(f"Input dimension: {input_dim}")
    n_actions = env.action_space.shape[0]               # for BipedalWalker-v3, n_actions = 4 in [-1, 1] range for each joint
    print(f"Number of actions: {n_actions}")
    
    model = ActorCritic(input_dim, n_actions)
    print("Model initialized.")
    print(model)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    rewards = train(env, model, optimizer, gamma=0.99, max_episodes=1000)
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Actor-Critic Training on CartPole')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig('actor_critic_cartpole_rewards.png')
    plt.show()
    
    # Save models
    torch.save(model.actor.state_dict(), 'actor_cartpole.pth')
    print("Actor model saved as actor_cartpole.pth")
    torch.save(model.critic.state_dict(), 'critic_cartpole.pth')
    print("Critic model saved as critic_cartpole.pth")
    torch.save(model.state_dict(), 'actor_critic_cartpole.pth')
    print("Actor-Critic model saved as actor_critic_cartpole.pth")
    
    
    # Test the trained model
    print("Testing trained model...")
    test_episodes = 10
    for episode in range(test_episodes):
        state = env.reset()
        if isinstance(state, tuple):  # Handle newer gym API
            state = state[0]
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.act(state)
            state, reward, done, info = env.step(action)[:4]
            total_reward += reward
            
        print(f"Test Episode {episode+1}: Reward = {total_reward}")

if __name__ == "__main__":
    main()
