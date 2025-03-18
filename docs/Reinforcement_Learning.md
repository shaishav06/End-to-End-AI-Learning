# Reinforcement Learning

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent takes actions, receives rewards or penalties, and adjusts its strategy to maximize long-term rewards.

## 1. Markov Decision Process (MDP)
MDP is a mathematical framework used to describe an RL problem. It consists of:

- **States (S)**: The set of all possible situations the agent can be in.
- **Actions (A)**: The set of all possible moves the agent can take.
- **Transition Probability (P)**: The probability of moving from one state to another given an action.
- **Reward Function (R)**: The reward received after taking an action in a given state.
- **Discount Factor (γ)**: A factor (0 ≤ γ ≤ 1) that determines the importance of future rewards.

**Example: Robot Navigation**
A robot in a grid world receives rewards for reaching a goal and penalties for hitting obstacles. The MDP helps define the best policy (set of actions) to maximize total rewards.

```python
import numpy as np

def transition(state, action):
    # Define transitions for a simple grid world
    if action == "up":
        return (state[0], state[1] + 1)
    elif action == "down":
        return (state[0], state[1] - 1)
    elif action == "left":
        return (state[0] - 1, state[1])
    elif action == "right":
        return (state[0] + 1, state[1])
    return state

# Example move
state = (1, 1)  # Current position
new_state = transition(state, "right")
print("New State:", new_state)
```

## 2. Q-Learning
Q-Learning is a model-free RL algorithm that helps an agent learn the optimal policy by updating a Q-table.

**Q-Table Update Rule:**
\[
Q(s, a) = Q(s, a) + \alpha [R + \gamma \max Q(s', a') - Q(s, a)]
\]
where:
- \( Q(s, a) \) is the value of action \( a \) in state \( s \).
- \( \alpha \) is the learning rate.
- \( \gamma \) is the discount factor.
- \( R \) is the reward received.
- \( \max Q(s', a') \) is the maximum future reward.

**Example: Q-Learning for a simple environment**
```python
import numpy as np
import random

# Initialize Q-table
actions = ["up", "down", "left", "right"]
Q_table = np.zeros((5, 5, len(actions)))  # 5x5 Grid

# Define parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)  # Explore
    else:
        return actions[np.argmax(Q_table[state[0], state[1]])]  # Exploit

def update_q_table(state, action, reward, next_state):
    action_index = actions.index(action)
    best_next_action = np.max(Q_table[next_state[0], next_state[1]])
    Q_table[state[0], state[1], action_index] += alpha * (reward + gamma * best_next_action - Q_table[state[0], state[1], action_index])
```

## 3. Deep Q-Networks (DQN)
DQN improves Q-Learning by using deep neural networks instead of a Q-table to approximate Q-values, making it scalable to large environments.

### Key Improvements:
- **Experience Replay**: Stores past experiences and samples them randomly to break correlation in training.
- **Target Network**: Uses a separate network to stabilize Q-value updates.

**Example: DQN for CartPole Game**
```python
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# Define DQN Model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

# Initialize environment
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Create DQN model
model = DQN(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop (simplified)
for episode in range(100):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    done = False
    while not done:
        q_values = model(state)
        action = torch.argmax(q_values).item()
        next_state, reward, done, _ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        target = reward + 0.99 * torch.max(model(next_state))
        loss = criterion(q_values[action], target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Conclusion
Reinforcement Learning enables agents to learn optimal behaviors through interaction with environments. MDP provides a mathematical foundation, Q-Learning allows tabular learning, and Deep Q-Networks extend it to complex tasks.

---
### [Back to Main README](../README.md)
