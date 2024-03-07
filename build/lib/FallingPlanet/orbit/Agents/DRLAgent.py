import gym
import torch
import torch.optim as optim
import random
from collections import deque
import numpy as np
from FallingPlanet.orbit.models.QNetworks import DCQN, DTQN
from torchvision import transforms
import time
import torch.nn.functional as F

class Agent:
    def __init__(self, env_name, policy_model, target_model, lr, gamma, epsilon_start, epsilon_end, n_episodes, memory_size,update_target_every):
        self.env = gym.make(env_name, render_mode=None)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Models
        self.policy_model = policy_model.to(self.device)
        self.target_model = target_model.to(self.device)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()  # Set the target network to evaluation mode
        self.UPDATE_TARGET_EVERY = update_target_every
        # Training Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_end / epsilon_start) ** (1 / n_episodes)
        self.optimizer = optim.RMSprop(self.policy_model.parameters(), lr=self.lr)
        
        # Experience Replay
        self.memory = deque(maxlen=memory_size)
        self.n_actions = self.env.action_space.n
        self.n_episodes = n_episodes

        # Metric Tracking
        self.metrics = {
            "rewards": [],
            "losses": [],
            "epsilon_values": [],
            "env_times": []
        }

        # Preprocessing Transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def update_target_network(self):
        """Update the weights of the target network to match the policy network."""
        self.target_model.load_state_dict(self.policy_model.state_dict())

    def update_epsilon(self):
        # Apply epsilon decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def preprocess_state(self, state):
        # Check if state is a tuple and has at least one element (the frame data)
        if isinstance(state, tuple) and len(state) > 0:
            # Extract the frame data (assuming it's the first element of the tuple)
            frame_data = state[0]
            
            # Now, preprocess frame_data as needed (e.g., resizing, grayscaling)
            if isinstance(frame_data, np.ndarray):
                processed_frames = []
                # Assuming frame_data is a single frame or stacked frames along the last dimension
                num_frames = frame_data.shape[-1] // 3  # Assuming 3 channels per frame
                
                for i in range(num_frames):
                    frame = frame_data[:, :, i*3:(i+1)*3]  # Extract ith frame
                    frame = self.transform(frame)  # Apply transformations defined in self.transform
                    processed_frames.append(frame)
                
                # Stack processed frames along the channel dimension to get [C, H, W]
                state_tensor = torch.cat(processed_frames, dim=0)
            else:
                raise TypeError("Frame data is not in the expected format.")
            
            # Add a batch dimension [B, C, H, W]
            state_tensor = state_tensor.unsqueeze(0).to(self.device)
            print(state_tensor.shape)
            return state_tensor
        else:
            raise TypeError("State is not in the expected format.")




    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Convert state to tensor, apply necessary transformations, etc.
        state_tensor = self.preprocess_state(state)
        
        # Use policy_model to get action values
        with torch.no_grad():
            action_values = self.policy_model(state_tensor)
        
        # Determine the action based on epsilon-greedy policy
        if random.random() > self.epsilon:
            return action_values.max(1)[1].item()  # Exploitation: choose best action
        else:
            return random.randrange(self.n_actions)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return  # Not enough samples to perform a replay

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Predicted Q-values for the current states
        current_q_values = self.policy_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute the maximum Q-values for the next states from the target network
        max_next_q_values = self.target_model(next_states).detach().max(1)[0]
        max_next_q_values[dones] = 0.0  # Zero out Q-values for terminal states

        # Compute target Q-values
        target_q_values = rewards + self.gamma * max_next_q_values

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)  # Decay epsilon

    def train(self, n_episodes, batch_size):
        for episode in range(n_episodes):
            start_time = time.time()
            state = self.env.reset()
            total_reward = 0
            terminated = False
            while not terminated:
                action = self.act(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.remember(state, action, reward, next_state, terminated or truncated)
                state = next_state
                total_reward += reward
                self.replay(batch_size)

            self.update_epsilon()
            self.metrics["rewards"].append(total_reward)
            self.metrics["epsilon_values"].append(self.epsilon)
            self.metrics["env_times"].append(time.time() - start_time)
            
            # Periodically update target network
            if episode % self.UPDATE_TARGET_EVERY == 0:
                self.update_target_network()
            print(f"Episode: {episode+1}, Total reward: {total_reward}, Epsilon: {self.epsilon}")


# Initialize environment and model
env = gym.make('ALE/Asteroids-v5')
n_actions = env.action_space.n
n_observation = 1  # Assuming a stack of 3 frames if not using frame stacking, adjust accordingly

# Instantiate policy and target models
policy_model = DCQN(n_observation=n_observation, n_actions=n_actions)
target_model = DCQN(n_observation=n_observation, n_actions=n_actions)  # Clone of policy model

# Instantiate the agent
n_episodes = 1000
memory_size = 100000
agent = Agent(env_name='ALE/Asteroids-v5', policy_model=policy_model, target_model=target_model, lr=3e-4, gamma=0.99, epsilon_start=1, epsilon_end=0.01, n_episodes=n_episodes, memory_size=memory_size,update_target_every=10)

# Start training
batch_size = 32
agent.train(n_episodes=n_episodes, batch_size=batch_size)


