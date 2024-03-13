import gym
import torch
import torch.optim as optim
import random
from collections import deque
import numpy as np
from FallingPlanet.orbit.models.QNetworks import DCQN, DTQN
from torchvision import transforms
import time
import torch
import torchrl
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gym.wrappers import FrameStack
import PIL.Image as Image
from torchrl.data.replay_buffers import TensorDictReplayBuffer, ReplayBuffer
from torchrl.data import LazyMemmapStorage
from torchrl.data import SliceSampler
from tensordict import TensorDict

class EfficientReplayBuffer:
    def __init__(self, capacity, state_shape, action_shape, reward_shape=(1,), done_shape=(1,), device="cuda"):
        self.device = device
        storage = LazyMemmapStorage(max_size=capacity)
        self.replay_buffer = TensorDictReplayBuffer(
            storage=storage,
            sampler=SliceSampler(num_slices=4),
            batch_size=32,  # Adjust based on your needs
        )
        # Initialize storage keys
        self.replay_buffer.extend({
            "states": torch.zeros((capacity, *state_shape), dtype=torch.float32, device=device),
            "actions": torch.zeros((capacity, *action_shape), dtype=torch.float32, device=device),
            "rewards": torch.zeros((capacity, *reward_shape), dtype=torch.float32, device=device),
            "next_states": torch.zeros((capacity, *state_shape), dtype=torch.float32, device=device),
            "dones": torch.zeros((capacity, *done_shape), dtype=torch.bool, device=device)
        })

    def add(self, state, action, reward, next_state, done):
        # Add experience to the replay buffer
        idx = self.replay_buffer.storage.size  # Get the next index to store data
        self.replay_buffer.storage.set(idx, {
            "states": state.to(self.device),
            "actions": action.to(self.device),
            "rewards": reward.to(self.device),
            "next_states": next_state.to(self.device),
            "dones": done.to(self.device)
        })

    def sample(self):
        # Sample a batch of experiences
        return self.replay_buffer.sample()

class Agent:
    def __init__(self, env_name, policy_model, target_model, lr, gamma, epsilon_start, epsilon_end, n_episodes, memory_size, update_target_every,frame_skip):
        self.env = gym.make(env_name, render_mode=None,full_action_space=False)
        self.env = FrameStack(self.env,4)
        self.env.seed(42)
        
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
        self.storage = LazyMemmapStorage(max_size=memory_size)
        self.replay_buffer = TensorDictReplayBuffer(storage=self.storage, batch_size=64)
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
        processed_frames = []
        
        # Check if state is a LazyFrames object
        if isinstance(state, gym.wrappers.frame_stack.LazyFrames):
            frames = state
        elif isinstance(state, tuple):
            # Assuming the first element of the tuple is what you need
            frames = state[0]
        else:
            raise ValueError("Unsupported state type received for preprocessing")
        
        # Now, 'frames' should be iterable and contain the stacked frames
        for frame in frames:
            frame_processed = self.process_single_frame(frame)
            processed_frames.append(frame_processed)
        
        # Stack processed frames along a new dimension and add a batch dimension
        stacked_frames = torch.stack(processed_frames, dim=0).unsqueeze(0)  # Shape: (1, 4, H, W)
        
        return stacked_frames.to(self.device)

    def process_single_frame(self, frame):
        """Process a single frame."""
        frame = np.array(frame).astype(np.uint8)  # Convert frame to uint8 numpy array if it's not already
        frame = np.mean(frame, axis=-1)  # Convert to grayscale
        pil_image = Image.fromarray(frame).convert("L")  # Convert to PIL Image and ensure grayscale
        transformed_frame = self.transform(pil_image)  # Apply transformations defined in self.transform
        
        return transformed_frame


    def remember(self, state, action, reward, next_state, done):
        # Preprocess state and next_state to ensure they have the correct shape
        state = self.preprocess_state(state)
        next_state = self.preprocess_state(next_state)

        # Explicitly determine the batch size from the state or next_state tensor
        batch_size = state.shape[0]  # Assuming the first dimension is the batch size

        # Now explicitly provide the batch size when creating the TensorDict
        experience = TensorDict({
            'states': state,
            'actions': torch.tensor([action], dtype=torch.long, device=self.device).unsqueeze(0),
            'rewards': torch.tensor([reward], dtype=torch.float32, device=self.device).unsqueeze(0),
            'next_states': next_state,
            'dones': torch.tensor([done], dtype=torch.bool, device=self.device).unsqueeze(0),
        }, batch_size=torch.Size([batch_size]))

        # Extend the replay buffer with the TensorDict
        self.replay_buffer.extend(experience)






    def act(self, state):
        state_tensor = self.preprocess_state(state)
        with torch.no_grad():
            action_values = self.policy_model(state_tensor)
        if random.random() > self.epsilon:
            return action_values.max(1)[1].item()
        else:
            return random.randrange(self.n_actions)

    def replay(self):
        # Ensure there are enough samples in the replay buffer for a batch
        if len(self.replay_buffer._storage) < self.replay_buffer._batch_size:
            return
        
        # Sample a batch of experiences
        sampled_experiences = self.replay_buffer.sample()  # Adjust this line if your method signature requires the batch size

        states = sampled_experiences["states"].to(self.device)
        actions = sampled_experiences["actions"].to(self.device)
        rewards = sampled_experiences["rewards"].to(self.device)
        next_states = sampled_experiences["next_states"].to(self.device)
        dones = sampled_experiences["dones"].to(self.device)

        # Debugging prints
        
        # The gather operation
        try:
            current_q_values = self.policy_model(states).gather(1, actions).squeeze(1)
        except RuntimeError as e:
            print("Error during .gather() operation:")
            print(e)
            # Additional debug prints if needed
            raise

        # Check actions are within expected range
        num_actions = self.policy_model(states).shape[1]  # This should correspond to the action space size
        assert actions.max() < num_actions, "Action index out of bounds"

        next_q_values = self.target_model(next_states).detach().max(1)[0]

        if dones.dim() > 1:
            dones = dones.squeeze()  # Or explicitly select the correct dimension if necessary

        # Ensure next_q_values has the same shape as current_q_values
        if next_q_values.dim() < 2:
            next_q_values = next_q_values.unsqueeze(1)

        # Adjust next_q_values based on dones
        next_q_values[dones] = 0.0

       

        target_q_values = rewards + self.gamma * next_q_values
        target_q_values = target_q_values.squeeze(-1)
        loss = F.huber_loss(current_q_values, target_q_values)
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
                for _ in range(self.frame_skip):
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    
                    cumulative_reward += reward

                    
                    cumulative_reward += reward
                    
                    frame_count += 1  # Increment frame count
                    if terminated or truncated:
                        break
                self.remember(state, action, cumulative_reward, next_state, terminated or truncated)
                state = next_state
                total_reward += cumulative_reward
                self.replay()

            self.update_epsilon()
            self.metrics["rewards"].append(total_reward)
            self.metrics["epsilon_values"].append(self.epsilon)
            self.metrics["env_times"].append(time.time() - start_time)
            self.metrics["frame_counts"].append(frame_count)  # Track frame count per episode

            if episode % self.UPDATE_TARGET_EVERY == 0:
                self.update_target_network()

            if episode % 500 == 0:  # Save the model every 500 episodes
                self.save_model(f"F:\FP_Agents\SpaceInvaders\dtqn2_policy_model_episode_{episode}.pth")

            print(f"Episode: {episode+1}, Total reward: {total_reward}, Epsilon: {self.epsilon}, Frames: {frame_count}, Loss: {self.metrics['losses'][-1] if self.metrics['losses'] else 'N/A'}")

            
    def save_model(self, filename="F:\FP_Agents\SpaceInvaders\dtqn2_policy_model.pth"):
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
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']  # Define a color scheme

    # Rewards per Episode
    plt.subplot(2, 2, 1)
    plt.plot(metrics["rewards"], color=colors[0], label='Rewards')
    plt.title("Rewards per Episode")
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    # Epsilon Decay
    plt.subplot(2, 2, 2)
    plt.plot(metrics["epsilon_values"], color=colors[1], label='Epsilon')
    plt.title("Epsilon Decay")
    plt.xlabel('Episode')
    plt.ylabel('Epsilon Value')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    # Loss per Episode
    plt.subplot(2, 2, 3)
    plt.plot(metrics["losses"], color=colors[2], label='Loss')
    plt.title("Loss per Frame")
    plt.xlabel('Frames')
    plt.ylabel('Loss')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    # Environment Time per Episode
    plt.subplot(2, 2, 4)
    plt.plot(metrics["env_times"], color=colors[3], label='Env Time')
    plt.title("Environment Time per Episode")
    plt.xlabel('Episode')
    plt.ylabel('Time (s)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()
        




if __name__ == '__main__':
    # Initialize environment and model
    env_name = "ALE/SpaceInvaders-v5"
    env = gym.make(env_name)
    env = FrameStack(env,4)
    n_actions = env.action_space.n
    
    n_observation = 6  # Assuming a stack of 3 frames if not using frame stacking, adjust accordingly

    # Instantiate policy and target models
    #policy_model = DCQN(n_observation=n_observation, n_actions=n_actions)
    #target_model = DCQN(n_observation=n_observation, n_actions=n_actions)  # Clone of policy model
    policy_model = DTQN(num_actions=n_observation, embed_size=256, num_heads=8, num_layers=4)  # Example values, adjust as needed
    target_model = DTQN(num_actions=n_observation, embed_size=256, num_heads=8, num_layers=4)
    # Instantiate the agent
    n_episodes = 10000
    memory_size = 100000
    agent = Agent(env_name=env_name, policy_model=policy_model, target_model=target_model, lr=1e-4, gamma=0.99, epsilon_start=1, epsilon_end=0.1, n_episodes=n_episodes, memory_size=memory_size, update_target_every=10, frame_skip=8)

    # Start training
    batch_size = 32
    agent.train(n_episodes=n_episodes, batch_size=batch_size)
    plot_metrics(agent.metrics)
    print(agent.metrics)


