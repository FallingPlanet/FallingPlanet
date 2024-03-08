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
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, env_name, policy_model, target_model, lr, gamma, epsilon_start, epsilon_end, n_episodes, memory_size, update_target_every,frame_skip):
        self.env = gym.make(env_name, render_mode="human",full_action_space=False)
        env.seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Models
        self.policy_model = policy_model.to(self.device)
        self.target_model = target_model.to(self.device)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()  # Set the target network to evaluation mode
        self.UPDATE_TARGET_EVERY = update_target_every
        # Training Hyperparameters
        self.frame_skip = frame_skip
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_end / epsilon_start) ** (1 / n_episodes)
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=self.lr)
        
        # Experience Replay
        self.memory = deque(maxlen=memory_size)
        self.n_actions = self.env.action_space.n
        self.n_episodes = n_episodes

        # Metric Tracking
        self.metrics = {
            "rewards": [],
            "losses": [],
            "epsilon_values": [],
            "env_times": [],
            "frame_counts":[]
        }

        # Define preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((110, 84)),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def update_target_network(self):
        self.target_model.load_state_dict(self.policy_model.state_dict())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)

    def preprocess_state(self, state):
        if not isinstance(state, np.ndarray) and (not isinstance(state, tuple) or not isinstance(state[0], np.ndarray)):
            raise TypeError("State must be a numpy array or a tuple containing a numpy array.")
        frame_data = state[0] if isinstance(state, tuple) else state
        if not (isinstance(frame_data, np.ndarray) and len(frame_data.shape) == 3 and frame_data.shape[2] == 3):
            raise TypeError("Frame data must be a numpy array with shape [Height, Width, Channels=3].")
        frame_data = np.mean(frame_data, axis=-1).astype(np.uint8)
        state_tensor = self.transform(frame_data)
        state_tensor = state_tensor.unsqueeze(0).to(self.device)
        return state_tensor

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state_tensor = self.preprocess_state(state)
        with torch.no_grad():
            action_values = self.policy_model(state_tensor)
        if random.random() > self.epsilon:
            return action_values.max(1)[1].item()
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

        target_q_values = rewards_t + self.gamma * next_q_values
        loss = F.mse_loss(current_q_values, target_q_values)
        self.metrics["losses"].append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, n_episodes, batch_size):
        for episode in range(n_episodes):
            start_time = time.time()
            state = self.env.reset()
            total_reward = 0
            terminated = False
            frame_count = 0  # Initialize frame counter for the episode
            
            while not terminated:
                action = self.act(state)
                cumulative_reward = 0
                for _ in range(self.frame_skip):  # Implement frame skipping
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    cumulative_reward += reward
                    frame_count += 1  # Increment frame count
                    if terminated or truncated:
                        break
                self.remember(state, action, cumulative_reward, next_state, terminated or truncated)
                state = next_state
                total_reward += cumulative_reward
                self.replay(batch_size)

            self.update_epsilon()
            self.metrics["rewards"].append(total_reward)
            self.metrics["epsilon_values"].append(self.epsilon)
            self.metrics["env_times"].append(time.time() - start_time)
            self.metrics["frame_counts"].append(frame_count)  # Track frame count per episode

            if episode % self.UPDATE_TARGET_EVERY == 0:
                self.update_target_network()

            if episode % 500 == 0:  # Save the model every 500 episodes
                self.save_model(f"policy_model_episode_{episode}.pth")

            print(f"Episode: {episode+1}, Total reward: {total_reward}, Epsilon: {self.epsilon}, Frames: {frame_count}, Loss: {self.metrics['losses'][-1] if self.metrics['losses'] else 'N/A'}")

            
    def save_model(self, filename="F:\FP_Agents\Asteroids\policy_model.pth"):
        """Save the model's state dict and other relevant parameters."""
        checkpoint = {
            'model_state_dict': self.policy_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'frame_skip': self.frame_skip  # Save frame_skip if you want it as part of your saved configuration
        }
        torch.save(checkpoint, filename)
        print(f"Model saved to {filename}")

def plot_metrics(metrics):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(metrics["rewards"])
    plt.title("Rewards per Episode")

    plt.subplot(2, 2, 2)
    plt.plot(metrics["epsilon_values"])
    plt.title("Epsilon Decay")

    plt.subplot(2, 2, 3)
    plt.plot(metrics["losses"])
    plt.title("Loss per Episode")

    plt.subplot(2, 2, 4)
    plt.plot(metrics["env_times"])
    plt.title("Environment Time per Episode")

    plt.tight_layout()
    plt.show()
        



# Initialize environment and model
env = gym.make('ALE/Asteroids-v5')
n_actions = env.action_space.n
n_observation = 1  # Assuming a stack of 3 frames if not using frame stacking, adjust accordingly

# Instantiate policy and target models
policy_model = DCQN(n_observation=n_observation, n_actions=n_actions)
target_model = DCQN(n_observation=n_observation, n_actions=n_actions)  # Clone of policy model

# Instantiate the agent
n_episodes = 1000
memory_size = 10000
agent = Agent(env_name='ALE/Asteroids-v5', policy_model=policy_model, target_model=target_model, lr=1e-2, gamma=0.99, epsilon_start=1, epsilon_end=0.01, n_episodes=n_episodes, memory_size=memory_size,update_target_every=5,frame_skip=8)

# Start training
batch_size = 32
agent.train(n_episodes=n_episodes, batch_size=batch_size)
plot_metrics(agent.metrics)


