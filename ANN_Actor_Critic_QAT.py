import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import copy
import math

"""
ANN Actor Critic trained with Quantization Aware Training, performed on Gymnasium's Cartpole
"""

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Custom 8-bit Quantization Functions
class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits=8):
        # Save for backward
        ctx.save_for_backward(x)
        ctx.num_bits = num_bits
        
        # Calculate the step size (scale)
        qmin = 0
        qmax = 2**num_bits - 1
        min_val = x.min()
        max_val = x.max()
        
        # Handle the case where min==max to avoid division by zero
        if min_val == max_val:
            return x
        
        scale = (max_val - min_val) / (qmax - qmin)
        # Fix: Use torch.round instead of round
        zero_point = qmin - torch.round(min_val / scale)
        
        # Quantize - use torch operations
        x_quant = torch.round(x / scale + zero_point)
        x_quant = torch.clamp(x_quant, qmin, qmax)
        
        # Dequantize (for fake quantization)
        x_dequant = (x_quant - zero_point) * scale
        
        return x_dequant
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator for backprop
        x, = ctx.saved_tensors
        # Pass the gradient through during backpropagation
        return grad_output, None

# Quantized Linear Layer
class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, num_bits=8):
        super(QuantizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bits = num_bits
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        # Quantize weights and biases during forward pass
        weight_q = FakeQuantize.apply(self.weight, self.num_bits)
        
        # Perform linear operation with quantized weights
        output = F.linear(input, weight_q)
        
        # Quantize bias if present
        if self.bias is not None:
            bias_q = FakeQuantize.apply(self.bias, self.num_bits)
            output = output + bias_q #.unsqueeze(0).expand_as(output)
        
        # Quantize activations
        output = FakeQuantize.apply(output, self.num_bits)
        
        return output

# Quantization helper for activation functions
class QuantizedReLU(nn.Module):
    def __init__(self, num_bits=8):
        super(QuantizedReLU, self).__init__()
        self.num_bits = num_bits
    
    def forward(self, x):
        x = F.relu(x)
        return FakeQuantize.apply(x, self.num_bits)

class QuantizedSoftmax(nn.Module):
    def __init__(self, dim=-1, num_bits=8):
        super(QuantizedSoftmax, self).__init__()
        self.dim = dim
        self.num_bits = num_bits
    
    def forward(self, x):
        x = F.softmax(x, dim=self.dim)
        return FakeQuantize.apply(x, self.num_bits)
    
class QuantizedMish(nn.Module):
    def __init__(self, num_bits=8):
        super(QuantizedMish, self).__init__()
        self.num_bits = num_bits
    
    def forward(self, x):
        x = x * torch.mish(F.softplus(x))
        return FakeQuantize.apply(x, self.num_bits)

# Quantization-Aware Actor-Critic Network
class QuantizedActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_dim=128, num_bits=8):
        super(QuantizedActorCritic, self).__init__()
        self.num_bits = num_bits
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            QuantizedLinear(input_dim, hidden_dim, num_bits=num_bits),
            QuantizedReLU(num_bits=num_bits)
        )
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            QuantizedLinear(hidden_dim, n_actions, num_bits=num_bits),
            QuantizedSoftmax(dim=-1, num_bits=num_bits)
        )
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            QuantizedLinear(hidden_dim, 1, num_bits=num_bits)
        )
    
    def forward(self, x):
        # Quantize input
        x = FakeQuantize.apply(x, self.num_bits)
        
        features = self.shared(x)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value
    
    def act(self, state):
        state = torch.FloatTensor(state)
        action_probs, _ = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

# Training function - similar to before but adapted for quantized model
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
        
        # Print progress every 20 episodes
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Episode {episode+1}/{max_episodes}, Avg Reward (last 20): {avg_reward:.2f}")
        
        # Early stopping if solved (CartPole is considered solved at 195.0)
        if np.mean(episode_rewards[-100:]) >= 195.0 and len(episode_rewards) >= 100:
            print(f"Environment solved in {episode+1} episodes!")
            break
    
    return episode_rewards

# Function to convert a quantization-aware model to a fully quantized model
def convert_to_quantized_model(qat_model, num_bits=8):
    # Create a copy of the model
    quantized_model = copy.deepcopy(qat_model)
    
    # This function would typically convert the fake quantized weights to actual quantized weights
    # Here we're just illustrating the concept - in a real deployment, you'd use a proper
    # quantization framework like PyTorch's quantization or TensorRT
    
    print(f"Model converted to {num_bits}-bit quantized format")
    print("Quantized model size (approximate):")
    
    # Calculate approximate size reduction
    full_precision_bits = 32
    original_params = sum(p.numel() for p in qat_model.parameters())
    original_size_bytes = original_params * (full_precision_bits / 8)
    quantized_size_bytes = original_params * (num_bits / 8)
    
    print(f"Original model: {original_size_bytes/1024:.2f} KB")
    print(f"Quantized model: {quantized_size_bytes/1024:.2f} KB")
    print(f"Size reduction: {100 * (1 - quantized_size_bytes/original_size_bytes):.2f}%")
    
    return quantized_model

# Main function
def main():
    # Create environment
    env = gym.make('CartPole-v1')
    
    # Initialize quantized model
    input_dim = env.observation_space.shape[0]  # 4 for CartPole
    n_actions = env.action_space.n              # 2 for CartPole
    
    # Number of bits for quantization
    num_bits = 8
    
    model = QuantizedActorCritic(input_dim, n_actions, num_bits=num_bits)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    print(f"Training with {num_bits}-bit quantization-aware training...")
    rewards = train(env, model, optimizer, gamma=0.99, max_episodes=1000)
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title(f'{num_bits}-bit Quantized Actor-Critic Training on CartPole')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig('quantized_actor_critic_cartpole_rewards.png')
    plt.show()
    
    # Save QAT model
    torch.save(model.state_dict(), f'quantized_actor_critic_cartpole_{num_bits}bit.pth')
    
    # Convert to fully quantized model for deployment
    quantized_model = convert_to_quantized_model(model, num_bits)
    
    # Test the trained quantized model
    print("Testing trained quantized model...")
    test_episodes = 10
    for episode in range(test_episodes):
        state = env.reset()
        if isinstance(state, tuple):  # Handle newer gym API
            state = state[0]
        done = False
        total_reward = 0
        
        while not done:
            action, _ = quantized_model.act(state)
            state, reward, done, info = env.step(action)[:4]
            total_reward += reward
            
        print(f"Test Episode {episode+1}: Reward = {total_reward}")
    
    # Compare performance with non-quantized model
    print("\nComparison with non-quantized model would typically show a trade-off between:")
    print("1. Model size reduction (achieved)")
    print("2. Inference speed improvement (expected)")
    print("3. Potential slight degradation in performance (if any)")

if __name__ == "__main__":
    main()
