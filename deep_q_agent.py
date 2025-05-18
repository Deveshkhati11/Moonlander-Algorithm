import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque


class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        # Enhanced neural network architecture with larger hidden layers
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),  # Increased from 64 to 128
            nn.ReLU(),
            nn.Linear(128, 128),  # Increased from 64 to 128
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)


# Priority item for experience replay
class PriorityItem:
    def __init__(self, state, action, reward, next_state, done, error=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.error = error if error is not None else 0.01  # Default small error


class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Neural networks
        self.policy_network = DeepQNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DeepQNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        # Improved hyperparameters
        self.learning_rate = learning_rate
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99  # Faster decay (was 0.995)
        self.tau = 0.01  # Soft update parameter

        # Double DQN flag
        self.use_double_dqn = True

        # Optimizer and loss function with gradient clipping
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Prioritized experience replay
        self.replay_memory = deque(maxlen=10000)
        self.batch_size = 64
        self.prioritized_replay = True
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4  # Importance sampling weight
        self.beta_increment = 0.001  # Increment beta each training step

        # Training metrics
        self.loss_history = []
        self.reward_history = []

    def select_action(self, state):
        # Epsilon-greedy action selection with decaying epsilon
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # Random action

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_network(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        # Calculate priority based on TD error
        if self.prioritized_replay:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

                current_q = self.policy_network(state_tensor)[0, action].item()
                next_q = self.target_network(next_state_tensor).max(1)[0].item()

                # Calculate TD error
                target = reward + (1 - done) * self.gamma * next_q
                error = abs(current_q - target) + 0.01  # Add small constant to avoid zero priority

                # Store with priority
                self.replay_memory.append(PriorityItem(state, action, reward, next_state, done, error))
        else:
            # Standard replay memory
            self.replay_memory.append(PriorityItem(state, action, reward, next_state, done))

    def train(self):
        # Check if enough samples are available
        if len(self.replay_memory) < self.batch_size:
            return

        # Sample batch with prioritization if enabled
        if self.prioritized_replay:
            # Get priorities
            priorities = np.array([item.error for item in self.replay_memory])
            priorities = priorities ** self.alpha

            # Calculate sampling probabilities
            probs = priorities / priorities.sum()

            # Sample based on priorities
            indices = np.random.choice(len(self.replay_memory), self.batch_size, p=probs)
            batch = [self.replay_memory[idx] for idx in indices]

            # Calculate importance sampling weights
            weights = (len(self.replay_memory) * probs[indices]) ** (-self.beta)
            weights /= weights.max()  # Normalize weights
            weights = torch.FloatTensor(weights).to(self.device)

            # Increase beta
            self.beta = min(1.0, self.beta + self.beta_increment)
        else:
            # Random sampling
            batch = random.sample(self.replay_memory, self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)

        # Extract batch data
        states = torch.FloatTensor([item.state for item in batch]).to(self.device)
        actions = torch.LongTensor([item.action for item in batch]).to(self.device)
        rewards = torch.FloatTensor([item.reward for item in batch]).to(self.device)
        next_states = torch.FloatTensor([item.next_state for item in batch]).to(self.device)
        dones = torch.FloatTensor([item.done for item in batch]).to(self.device)

        # Compute current Q values
        current_q_values = self.policy_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values with Double DQN
        if self.use_double_dqn:
            # Select actions using policy network
            next_actions = self.policy_network(next_states).max(1)[1].unsqueeze(1)
            # Evaluate actions using target network
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
        else:
            # Standard DQN
            next_q_values = self.target_network(next_states).max(1)[0]

        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute weighted loss for prioritized replay
        td_errors = torch.abs(current_q_values - target_q_values.detach())
        loss = (weights * td_errors ** 2).mean()

        # Store loss for monitoring
        self.loss_history.append(loss.item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities in replay memory for sampled experiences
        if self.prioritized_replay:
            for idx, error in zip(indices, td_errors.detach().cpu().numpy()):
                self.replay_memory[idx].error = error + 0.01  # Add small constant

        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        # Soft update of target network
        for target_param, policy_param in zip(self.target_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        """Save the trained model"""
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)

    def load(self, filename):
        """Load a trained model"""
        checkpoint = torch.load(filename)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']