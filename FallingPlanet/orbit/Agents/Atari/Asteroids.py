import gym
from FallingPlanet.orbit.models.QNetworks import DCQN, DTQN
from FallingPlanet.orbit.Agents.DRLAgent import Agent



# Instantiate the environment
env = gym.make('ALE/Asteroids-v5')

# Now we can access n_actions directly
n_actions = env.action_space.n

# For DCQN, we assume a fixed observation space size or reshape input accordingly
# Observation space dimensions should match your model's input layer
n_observation = 3  # Example shape for an RGB image, adjust based on your preprocessing

# Initialize the model
model = DCQN(n_observation=n_observation, n_actions=n_actions)
# Initialize the agent with the correct parameters
agent = Agent(env_name='ALE/Asteroids-v5', model=model, lr=1e-4, gamma=0.99, 
              epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, memory_size=10000)

# Start training
n_episodes = 500
batch_size = 32
agent.train(n_episodes=n_episodes, batch_size=batch_size)