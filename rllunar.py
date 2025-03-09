import gymnasium as gym
import numpy as np
import cv2
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import torch

#Video
# occlusion
# dust
# artifacts
# temporal issues
# acceleration limits
# work with different lunar environments =

# =======================
# Custom Gym Environment
# =======================
class LunarTerrainEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self):
        super(LunarTerrainEnv, self).__init__()
        # Define action space (0=Forward, 1=Left, 2=Right, 3=Hover)
        self.action_space = spaces.Discrete(4)
        # Define observation space (x, y, vx, vy, lidar1, lidar2)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.reset()

    def step(self, action):
        # Apply action effects
        if action == 0:  # Move Forward
            self.x += 0.1 * np.cos(self.angle)
            self.y += 0.1 * np.sin(self.angle)
        elif action == 1:  # Turn Left
            self.angle -= 0.1
        elif action == 2:  # Turn Right
            self.angle += 0.1
        elif action == 3:  # Hover (No Movement)
            pass

        # Compute reward and termination flag
        reward, done = self._calculate_reward()

        # Get sensor readings (fake lidar)
        sensors = self._get_lidar_readings()

        # Note: vx, vy are placeholders; you could update them if desired.
        obs = np.array([self.x, self.y, self.vx, self.vy] + sensors, dtype=np.float32)
        # Gymnasium step returns: observation, reward, done, truncated, info
        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        # Reset state
        self.x, self.y = 0.0, 0.0  # Starting position
        self.vx, self.vy = 0.0, 0.0  # Initial velocities
        self.angle = 0.0  # Facing right (0 rad)
        self.goal = np.array([4.5, 4.5])  # Goal position
        self.obstacles = np.random.uniform(0.5, 4.5, size=(5, 2))  # Regenerate obstacles
        # Return initial observation and info dict per Gymnasium API
        obs = np.array([self.x, self.y, self.vx, self.vy] + self._get_lidar_readings(), dtype=np.float32)
        return obs, {}

    def _calculate_reward(self):
        # Check for collisions
        if self._collided_with_obstacle():
            return -100, True  # Large penalty and episode termination
        # Check if goal reached
        if np.linalg.norm(np.array([self.x, self.y]) - self.goal) < 0.5:
            return 100, True  # Reward for reaching goal
        # Otherwise, a small step penalty to encourage faster solutions
        return -1, False

    def _get_lidar_readings(self):
        # Fake lidar: measure distance to each obstacle (cap distance at 1.0)
        distances = []
        for obs in self.obstacles:
            dist = np.linalg.norm(np.array([self.x, self.y]) - obs)
            distances.append(min(dist, 1.0))
        # Return the first two readings for simplicity (padding if needed)
        if len(distances) < 2:
            distances += [1.0] * (2 - len(distances))
        return distances[:2]

    def _collided_with_obstacle(self):
        for obs in self.obstacles:
            if np.linalg.norm(np.array([self.x, self.y]) - obs) < 0.3:
                return True
        return False

# ====================================
# Train RL Model using Stable-Baselines3 on GPU
# ====================================
def train_model():
    # Create a vectorized environment for training
    env = make_vec_env(LunarTerrainEnv, n_envs=4)
    # Ensure a GPU is available; set device to "cuda" for GPU training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = PPO("MlpPolicy", env, device=device, verbose=1)
    model.learn(total_timesteps=50000)
    model.save("lunar_rl_model")
    print("Training Complete. Model Saved.")

# =========================
# OpenCV Visualization
# =========================
class LunarTerrainOpenCV:
    def __init__(self, env, size=500):
        self.env = env
        self.size = size  # Size of the OpenCV window (in pixels)
        self.scale = size / 5  # Convert environment units (0-5) to pixels

    def render(self, obs):
        # Create a white background image
        img = np.ones((self.size, self.size, 3), dtype=np.uint8) * 255

        # Draw goal as a green circle
        goal_pos = (int(self.env.goal[0] * self.scale), int(self.env.goal[1] * self.scale))
        cv2.circle(img, goal_pos, 10, (0, 255, 0), -1)

        # Draw obstacles as black rectangles
        for o in self.env.obstacles:
            top_left = (int(o[0] * self.scale - 10), int(o[1] * self.scale - 10))
            bottom_right = (int(o[0] * self.scale + 10), int(o[1] * self.scale + 10))
            cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), -1)

        # Draw the agent as a red circle
        agent_pos = (int(obs[0] * self.scale), int(obs[1] * self.scale))
        cv2.circle(img, agent_pos, 10, (0, 0, 255), -1)
        return img

    def run(self, model_path="lunar_rl_model"):
        # Load the trained model
        model = PPO.load(model_path)
        obs, _ = self.env.reset()
        
        while True:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = self.env.step(action)
            frame = self.render(obs)
            cv2.imshow("Lunar Navigation", frame)
            # Wait 100ms; exit if 'q' is pressed
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            if done:
                # Reset environment if episode is over
                obs, _ = self.env.reset()
        cv2.destroyAllWindows()

# =========================
# Main Script Entry Point
# =========================
if __name__ == "__main__":
    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Device Count:", torch.cuda.device_count())
    print("Current CUDA Device:", torch.cuda.current_device())
    print("CUDA Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA Device")
    
    mode = input("Enter 'train' to train the model, or 'run' to visualize: ").strip().lower()

    if mode == "train":
        train_model()
    elif mode == "run":
        # Use a single environment instance for visualization
        env = LunarTerrainEnv()
        visualizer = LunarTerrainOpenCV(env)
        visualizer.run()
    else:
        print("Invalid choice. Please enter 'train' or 'run'.")
