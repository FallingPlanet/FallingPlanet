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
        self.env = gym.make(env_name, render_mode="human")
        self.env.metadata['render_fps']=120
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

        # Define preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.Resize((110, 84)),  # Resize to intermediate size
            transforms.CenterCrop(84),  # Crop the center 84x84 portion
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize pixel values
        ])

    def update_target_network(self):
        """Update the weights of the target network to match the policy network."""
        self.target_model.load_state_dict(self.policy_model.state_dict())

    def update_epsilon(self):
        # Apply epsilon decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def preprocess_state(self, state):
        # Ensure the state is in an expected container format (e.g., numpy array or a tuple containing a numpy array)
        if not isinstance(state, np.ndarray) and (not isinstance(state, tuple) or not isinstance(state[0], np.ndarray)):
            raise TypeError("State must be a numpy array or a tuple containing a numpy array.")

        # Extract frame data from state if it's in a tuple
        frame_data = state[0] if isinstance(state, tuple) else state

        # Verify frame data is a numpy array with the expected shape (H, W, C)
        if not (isinstance(frame_data, np.ndarray) and len(frame_data.shape) == 3 and frame_data.shape[2] == 3):
            raise TypeError("Frame data must be a numpy array with shape [Height, Width, Channels=3].")

        # Convert frame data to grayscale and ensure it's in uint8 format for PIL compatibility
        frame_data = np.mean(frame_data, axis=-1).astype(np.uint8)

        # Convert frame_data to PIL Image, apply transformations, and convert back to tensor
        state_tensor = self.transform(frame_data)

        # Add a batch dimension and transfer to the correct device
        state_tensor = state_tensor.unsqueeze(0).to(self.device)

        return state_tensor





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
            return

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states_processed = torch.cat([self.preprocess_state(state) for state in states])
        next_states_processed = torch.cat([self.preprocess_state(next_state) for next_state in next_states])


        actions_t = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.bool).to(self.device)

        current_q_values = self.policy_model(states_processed).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states_processed).detach().max(1)[0]
        next_q_values[dones_t] = 0.0

        # Compute target Q-values
        target_q_values = rewards_t + self.gamma * next_q_values

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

            # Ensure this print statement is within the for-loop
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


