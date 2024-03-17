import os
import torch
import gym
from gym.wrappers import FrameStack
from FallingPlanet.orbit.models.QNetworks import DTQN
import numpy as np
import PIL.Image as Image
from torchvision import transforms

class ModelEvaluator:
    def __init__(self, env_name, checkpoint_dir, num_eval_episodes=25):
        self.env_name = env_name
        self.checkpoint_dir = checkpoint_dir
        self.num_eval_episodes = num_eval_episodes
        self.env = gym.make(env_name, render_mode=None,full_action_space=False)
        self.env = FrameStack(self.env,4)
        self.env.seed(42)
        
    def load_model(self, checkpoint_path, model_class, n_observation, n_actions):
        model = model_class(num_actions=n_observation, embed_size=256, num_heads=8, num_layers=4)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def process_single_frame(self, frame):
        frame = np.squeeze(frame)  # Remove single-dimensional entries from the shape
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        pil_image = Image.fromarray(frame).convert("L")
        return pil_image


    def preprocess_state(self, state):
        """Preprocess state (collection of frames)."""
        processed_frames = [self.process_single_frame(frame) for frame in state]
        stacked_frames = torch.stack(processed_frames).unsqueeze(0)  # Add a batch dimension at the start
        return stacked_frames

    def evaluate_model(self, model):
        total_rewards = []
        for episode in range(self.num_eval_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                processed_state = self.preprocess_state(state)
                action_values = model(processed_state)
                action = action_values.max(1)[1].item()
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
            total_rewards.append(total_reward)
        return total_rewards

    def run_evaluation(self, model_class, n_observation, n_actions):
        checkpoints = [os.path.join(self.checkpoint_dir, f) for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth')]
        for checkpoint_path in checkpoints:
            model = self.load_model(checkpoint_path, model_class, n_observation, n_actions)
            total_rewards = self.evaluate_model(model)
            avg_reward = sum(total_rewards) / len(total_rewards)
            print(f"Checkpoint: {checkpoint_path}, Average Reward: {avg_reward}")

if __name__ == '__main__':
    evaluator = ModelEvaluator(
        env_name="ALE/SpaceInvaders-v5",
        checkpoint_dir="F:/FP_Agents/SpaceInvaders/",
        num_eval_episodes=25
    )
    
    # Assuming DTQN is your model class and passing the required arguments
    evaluator.run_evaluation(DTQN, n_observation=6, n_actions=6)
