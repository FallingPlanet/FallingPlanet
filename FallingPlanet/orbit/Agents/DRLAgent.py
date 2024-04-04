import gym
import torch
import torch.optim as optim
import random
from collections import deque
import numpy as np
from FallingPlanet.orbit.models.QNetworks import DCQN, DTQN, EfficientAttentionModel
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
import sys
import os
from torch.utils.tensorboard import SummaryWriter


class SequentialReplayBuffer:
    def __init__(self, sequence_length, batch_size, buffer_size, state_shape, action_dim, device):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.buffer_size = buffer_size // sequence_length
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.device = device
        
        self.buffer = deque(maxlen=self.buffer_size)
        self.current_sequence = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}
        
    def _add_to_sequence(self, state, action, reward, next_state, done):
        self.current_sequence['states'].append(state)
        self.current_sequence['actions'].append(action)
        self.current_sequence['rewards'].append(reward)
        self.current_sequence['next_states'].append(next_state)
        self.current_sequence['dones'].append(done)
        
        if len(self.current_sequence['states']) == self.sequence_length:
            self.buffer.append({k: np.array(v) for k, v in self.current_sequence.items()})
            self.current_sequence = {k: [] for k in self.current_sequence}
    
    def add(self, state, action, reward, next_state, done):
        self._add_to_sequence(state, action, reward, next_state, done)
    
    def sample(self):
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        sampled_sequences = [self.buffer[idx] for idx in indices]
        
        batch = {k: torch.tensor(np.concatenate([seq[k] for seq in sampled_sequences]), dtype=torch.float32).to(self.device) for k in self.current_sequence}
        
        # Convert actions, rewards, and dones to appropriate types
        batch['actions'] = batch['actions'].long()
        batch['rewards'] = batch['rewards'].float()
        batch['dones'] = batch['dones'].float()
        
        return batch
    
    def __len__(self):
        return len(self.buffer) * self.sequence_length

class Agent:
    def __init__(self, env_name, policy_model, target_model, lr, gamma, epsilon_start, epsilon_end, n_episodes, memory_size, update_target_every,frame_skip,buffer_type = None):
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
        if buffer_type == "sequential":
          
            self.memory = SequentialReplayBuffer(sequence_length=4, batch_size=64, buffer_size=memory_size, state_shape=env.observation_space.shape, action_dim=env.action_space.n, device=self.device)
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
        loss = F.mse_loss(current_q_values, target_q_values)
        self.metrics["losses"].append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        
        
        self.optimizer.step()

    def train(self, n_episodes, batch_size):
        writer = SummaryWriter('runs/DCQN_10k_SpaceInvaders')

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
            # Log metrics to TensorBoard
            writer.add_scalar('Reward', total_reward, episode)
            writer.add_scalar('Epsilon', self.epsilon, episode)
            writer.add_scalar('Loss', self.metrics['losses'][-1] if self.metrics['losses'] else 0, episode)
            writer.add_scalar('Env Time', time.time() - start_time, episode)
            
            if episode % self.UPDATE_TARGET_EVERY == 0:
                self.update_target_network()

            if episode % 500 == 0:  # Save the model every 500 episodes
                self.save_model(f"F:\FP_Agents\SpaceInvaders\dcqn\_policy_model_episode_{episode}.pth")
            if episode == 100000:
                self.save_model(f"F:\FP_Agents\SpaceInvaders\dcqn\_policy_model_episode_{episode}.pth")
            # Periodic evaluation
            if episode % 100 == 0 and episode > 0:  # Avoid evaluation at the very start
                avg_reward = self.evaluate(n_eval_episodes=5)  # Adjust n_eval_episodes as needed
                writer.add_scalar('Evaluation/Average Reward', avg_reward, episode)
                print(f"Evaluation after episode {episode}: Average Reward = {avg_reward}")
                
            print(f"Episode: {episode+1}, Total reward: {total_reward}, Epsilon: {self.epsilon}, Frames: {frame_count}, Loss: {self.metrics['losses'][-1] if self.metrics['losses'] else 'N/A'}")

            
    def save_model(self, filename="F:\FP_Agents\SpaceInvaders\dcqn\_policy_model.pth"):
        """Save the model's state dict and other relevant parameters."""
        checkpoint = {
            'model_state_dict': self.policy_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'frame_skip': self.frame_skip  # Save frame_skip if you want it as part of your saved configuration
        }
        torch.save(checkpoint, filename)
        print(f"Model saved to {filename}")
    def evaluate(self, n_eval_episodes=25):
            total_rewards = []
            for episode in range(n_eval_episodes):
                state = self.env.reset()
                total_reward = 0
                done = False
                while not done:
                    state_tensor = self.preprocess_state(state)  # Ensure state is preprocessed
                    with torch.no_grad():
                        action_values = self.policy_model(state_tensor)
                        action = action_values.max(1)[1].item()  # Choose the best action
                    state, reward, terminated,truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                total_rewards.append(total_reward)
                print(f"Eval Episode {episode+1}/{n_eval_episodes}: Total Reward = {total_reward}")
            avg_reward = sum(total_rewards) / len(total_rewards)
            print(f"Average Reward over {n_eval_episodes} episodes: {avg_reward}")
            return avg_reward
        
    def evaluate_with_checkpoint(self, checkpoint_path):
        # Load the checkpoint into the model
        checkpoint = torch.load(checkpoint_path)
        self.policy_model.load_state_dict(checkpoint['model_state_dict'])
        self.policy_model.eval()  # Ensure the model is in evaluation mode

        # Evaluate the model
        avg_reward = self.evaluate(n_eval_episodes=5)
        print(f"Evaluated {checkpoint_path}: Average Reward = {avg_reward}")
        
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
    mode = "train"  # Default mode
    if len(sys.argv) > 1:
        mode = sys.argv[1]  # Assume the second argument specifies mode
    # Initialize environment and model
    env_name = "ALE/SpaceInvaders-v5"
    env = gym.make(env_name)
    env = FrameStack(env,4)
    n_actions = env.action_space.n
    
    n_observation = 18 # Assuming a stack of 3 frames if not using frame stacking, adjust accordingly

    # Instantiate policy and target models
    policy_model = DCQN(n_actions=n_actions)
    target_model = DCQN(n_actions=n_actions)  # Clone of policy model
    #policy_model = DTQN(num_actions=n_observation, embed_size=512, num_heads=16, num_layers=3,patch_size=16)  # Example values, adjust as needed
    #target_model = DTQN(num_actions=n_observation, embed_size=512, num_heads=16, num_layers=3,patch_size=16)
    #policy_model = EfficientAttentionModel(num_actions=6,input_dim=84*84*4,embed_size=512,num_layers=5)
    #target_model = EfficientAttentionModel(num_actions=6,input_dim=84*84*4,embed_size=512,num_layers=5)
    # Instantiate the agent
    n_episodes = 10001
    memory_size = 100000
    agent = Agent(env_name=env_name, policy_model=policy_model, target_model=target_model, lr=1e-3, gamma=0.99, epsilon_start=1, epsilon_end=0.1, n_episodes=n_episodes, memory_size=memory_size, update_target_every=50, frame_skip=4,buffer_type="efficient")

    # Start training
    batch_size = 32
   
    checkpoint_dir = "F:\FP_Agents\SpaceInvaders\dcqn"
    if mode == "train":
        print("Starting Training...")
        agent.train(n_episodes=n_episodes, batch_size=32)
        plot_metrics(agent.metrics)
    elif mode == "eval":
        print("Starting Evaluation...")
        # Iterate through each checkpoint in the directory and evaluate it
        for checkpoint_filename in sorted(os.listdir(checkpoint_dir)):
            if checkpoint_filename.endswith('.pth'):
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
                agent.evaluate_with_checkpoint(checkpoint_path)
    else:
        print(f"Unknown mode: {mode}. Please use 'train' or 'eval'.")

