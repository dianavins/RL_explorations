import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from utils.record_saving import *

"""
Basic ANN Actor Critic model, performed on Gymnasium's Cartpole
"""

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.set_default_device(device)

# Define the Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # Actor head (policy network) - outputs action probabilities
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value network) - outputs state value
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value
    
    def act(self, state):
        state = torch.FloatTensor(state)
        action_probs, _ = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

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
            action, log_prob = model.act(state)
            # Take action
            next_state, reward, done, info = env.step(action)[:4]
            
            # Store transition
            state_tensor = torch.FloatTensor(state)
            _, value = model(state_tensor)
            
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
        
        # print("episode:", episode, "reward:", episode_reward)
        
        # Print progress every 10 episodes
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Episode {episode+1}/{max_episodes}, Avg Reward (last 20): {avg_reward:.2f}")
        
        # Early stopping if solved (CartPole is considered solved at 195.0)
        if np.mean(episode_rewards[-100:]) >= 195.0 and len(episode_rewards) >= 100:
            print(f"Environment solved in {episode+1} episodes!")
            break
    
    return episode_rewards, episode+1

# Main function
def main():
    # Create environment
    env = gym.make('CartPole-v1')
    # env = gym.make("MountainCar-v0", render_mode="rgb_array", goal_velocity=0.1)
    # env = gym.make('Acrobot-v1', render_mode="rgb_array")
    print("Environment created.")
    
    # Initialize model
    input_dim = env.observation_space.shape[0]  # 4 for CartPole
    n_actions = env.action_space.n               # 2 for CartPole
    model = ActorCritic(input_dim, n_actions)
    print("Model initialized.")
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    rewards, num_episodes = train(env, model, optimizer, gamma=0.99, max_episodes=1000)
    
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
        
    # Save training results
    avg_reward_last_20 = np.mean(rewards[-20:])
    log_training_results("Actor-Critic", "unquantized", "CartPole-v1", num_episodes, avg_reward_last_20)

if __name__ == "__main__":
    main()
